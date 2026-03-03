#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

# PyG
from torch_geometric.nn import SAGEConv

# --------------------------
# Utils (KEEP SAME)
# --------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pearsonr_np(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum())) + eps
    return float((a*b).sum() / denom)

def r2_np(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-8) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum() + eps
    return float(1.0 - ss_res / ss_tot)

def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))

def log(msg: str, fp=None):
    print(msg, flush=True)
    if fp is not None:
        fp.write(msg + "\n")
        fp.flush()

# --------------------------
# Split: spatial block split (grid) (KEEP SAME)
# --------------------------
def make_block_split(pos: np.ndarray,
                     grid: Tuple[int,int]=(6,6),
                     test_frac: float=0.2,
                     seed: int=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    pos: (N,2) array
    returns train_mask, test_mask
    Strategy:
      - bin points into grid cells
      - sample ~test_frac of cells as test cells
      - test spots are those in test cells
    """
    assert pos.ndim == 2 and pos.shape[1] == 2
    N = pos.shape[0]
    gx, gy = grid

    x = pos[:,0]
    y = pos[:,1]
    # normalize to [0,1]
    x01 = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y01 = (y - y.min()) / (y.max() - y.min() + 1e-8)

    ix = np.clip((x01 * gx).astype(int), 0, gx-1)
    iy = np.clip((y01 * gy).astype(int), 0, gy-1)
    cell = ix * gy + iy  # 0..gx*gy-1

    n_cells = gx * gy
    rng = np.random.RandomState(seed)
    cells = np.arange(n_cells)
    rng.shuffle(cells)
    n_test_cells = max(1, int(round(n_cells * test_frac)))
    test_cells = set(cells[:n_test_cells].tolist())

    test_mask = np.array([c in test_cells for c in cell], dtype=bool)
    # avoid degenerate
    if test_mask.mean() < 0.05 or test_mask.mean() > 0.5:
        # fallback: adjust to nearest reasonable by random spot split (rare)
        rng = np.random.RandomState(seed+999)
        test_mask = rng.rand(N) < test_frac

    train_mask = ~test_mask
    return train_mask, test_mask

# --------------------------
# Normalization: log1p + train z-score (KEEP SAME)
# --------------------------
@dataclass
class YNorm:
    mean: np.ndarray
    std: np.ndarray

def normalize_y(y: np.ndarray, train_mask: np.ndarray, eps=1e-6) -> Tuple[np.ndarray, YNorm]:
    """
    y: (N,P) raw pathway values
    returns y_norm and norm stats (mean/std computed on train only)
    """
    y1 = np.log1p(y.astype(np.float32))
    mu = y1[train_mask].mean(axis=0)
    sd = y1[train_mask].std(axis=0)
    sd = np.clip(sd, eps, None)
    y_norm = (y1 - mu) / sd
    return y_norm, YNorm(mean=mu, std=sd)

def denormalize_y(y_norm: np.ndarray, norm: YNorm) -> np.ndarray:
    y1 = y_norm * norm.std + norm.mean
    y = np.expm1(y1)
    return y

# ============================================================
# NEW: Pathway graph utilities (DualGraph only)
# ============================================================
def load_pathway_graph_A_norm(pathways: List[str], edge_tsv: str, device: torch.device):
    """
    Build normalized adjacency A_norm (P×P) sparse:
      A_norm = D^{-1/2} (A + I) D^{-1/2}
    edge_tsv: two columns: pathway_a \t pathway_b (undirected)
    """
    pid2i = {p:i for i,p in enumerate(pathways)}
    src, dst = [], []

    with open(edge_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            if (a in pid2i) and (b in pid2i):
                ia, ib = pid2i[a], pid2i[b]
                # undirected
                src += [ia, ib]
                dst += [ib, ia]

    P = len(pathways)
    # self loops
    for i in range(P):
        src.append(i)
        dst.append(i)

    idx = torch.tensor([src, dst], dtype=torch.long, device=device)
    val = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    A = torch.sparse_coo_tensor(idx, val, (P, P)).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min(1.0)
    d_inv_sqrt = deg.pow(-0.5)
    r, c = A.indices()
    v = A.values() * d_inv_sqrt[r] * d_inv_sqrt[c]
    A_norm = torch.sparse_coo_tensor(A.indices(), v, (P, P)).coalesce()
    return A_norm

class PathwayGraphHead(nn.Module):
    """
    Apply pathway coupling on output dimension:
      Z_smooth = Z @ A_norm
      Y = alpha * Z_smooth + (1-alpha) * Z
    alpha is learnable scalar in (0,1).
    """
    def __init__(self, A_norm_sparse: torch.Tensor, learnable_alpha: bool=True):
        super().__init__()
        self.register_buffer("A_norm", A_norm_sparse)
        self.logit_alpha = nn.Parameter(torch.tensor(0.0)) if learnable_alpha else None

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: (N,P); want Z @ A_norm (P×P sparse)
        # do transpose trick: (P×P) @ (P×N) = (P×N)
        ZT = Z.transpose(0, 1).contiguous()           # P×N
        Zs_T = torch.sparse.mm(self.A_norm, ZT)       # P×N
        Zs = Zs_T.transpose(0, 1).contiguous()        # N×P

        if self.logit_alpha is None:
            return Zs
        alpha = torch.sigmoid(self.logit_alpha)
        return alpha * Zs + (1.0 - alpha) * Z

# --------------------------
# Models (KEEP MLP & GraphSAGE SAME; ADD DualGraph)
# --------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int=512, dropout: float=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int=256, dropout: float=0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.head(x)
        return x

class DualGraphSAGE(nn.Module):
    """
    IMPORTANT:
      - Spot encoder part is IDENTICAL to GraphSAGE up to conv2 output.
      - Then predict Z (N×P) same as GraphSAGE head would do,
        and apply PathwayGraphHead to couple pathways.
    This ensures GNN baseline stays unchanged; DualGraph is an extra branch.
    """
    def __init__(self, in_dim: int, out_dim: int, A_norm_sparse: torch.Tensor,
                 hidden: int=256, dropout: float=0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        # keep same head structure as GraphSAGE
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
        self.dropout = dropout
        self.path_head = PathwayGraphHead(A_norm_sparse, learnable_alpha=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        z = self.head(x)            # N×P (same as GraphSAGE output head)
        y = self.path_head(z)       # N×P coupled
        return y

# --------------------------
# Train / Eval (KEEP SAME)
# --------------------------
def train_one(model, optimizer, x, edge_index, y, train_mask, device, is_gnn: bool):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if is_gnn:
        pred = model(x, edge_index)
    else:
        pred = model(x)
    loss = F.mse_loss(pred[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())

@torch.no_grad()
def predict(model, x, edge_index, device, is_gnn: bool) -> torch.Tensor:
    model.eval()
    if is_gnn:
        return model(x, edge_index)
    else:
        return model(x)

def compute_metrics_per_pathway(gt: np.ndarray, pred: np.ndarray, test_mask: np.ndarray, colnames) -> pd.DataFrame:
    """
    gt/pred are in ORIGINAL SCALE (not z-score), shape (N,P)
    metrics computed on test spots only, per pathway
    """
    gt_t = gt[test_mask]
    pr_t = pred[test_mask]
    rows = []
    for j, name in enumerate(colnames):
        g = gt_t[:, j]
        p = pr_t[:, j]
        rows.append({
            "pathway": name,
            "pearson": pearsonr_np(g, p),
            "r2": r2_np(g, p),
            "rmse": rmse_np(g, p),
            "gt_mean": float(g.mean()),
            "gt_nz_ratio": float((g != 0).mean())
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", required=True)
    ap.add_argument("--y_dir", required=True)
    ap.add_argument("--patient", required=True, help="e.g. Y27")
    ap.add_argument("--y_suffix", default="_T_SM_Pathway.filtered.spot.intersect.csv")

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid", type=str, default="6,6")
    ap.add_argument("--test_frac", type=float, default=0.2)

    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_mlp", type=int, default=512)
    ap.add_argument("--hidden_gnn", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--early_patience", type=int, default=80)

    # NEW: pathway graph edges
    ap.add_argument("--pathway_edges_tsv", default="",
                    help="If provided, run DualGraph using this pathway_edges.tsv")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for sub in ["ckpt", "metrics", "preds", "logs", "cache"]:
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    log_fp = os.path.join(args.out_dir, "logs", f"{args.patient}.log")
    fplog = open(log_fp, "w", encoding="utf-8")

    set_seed(args.seed)
    grid = tuple(map(int, args.grid.split(",")))

    # load
    graph_fp = os.path.join(args.graph_dir, f"{args.patient}.pt")
    y_fp = os.path.join(args.y_dir, f"{args.patient}{args.y_suffix}")

    log(f"[{args.patient}] graph: {graph_fp}", fplog)
    log(f"[{args.patient}] y    : {y_fp}", fplog)

    # IMPORTANT: PyTorch>=2.6 must load weights_only=False for PyG Data
    data = torch.load(graph_fp, map_location="cpu", weights_only=False)
    y_df = pd.read_csv(y_fp, index_col=0)

    # assert alignment
    g_spots = list(map(str, data.spot_id))
    y_spots = list(y_df.index.astype(str))
    assert g_spots == y_spots, "spot_id order mismatch (should already be aligned)"

    x = data.x.float()
    pos = data.pos.detach().cpu().numpy()
    edge_index = data.edge_index.long()

    y = y_df.values.astype(np.float32)
    colnames = list(y_df.columns)

    # split
    train_mask, test_mask = make_block_split(pos, grid=grid, test_frac=args.test_frac, seed=args.seed)
    log(f"split: grid={grid}, test_frac={args.test_frac}, train={train_mask.sum()}, test={test_mask.sum()}", fplog)

    # normalize y
    y_norm, norm = normalize_y(y, train_mask)
    y_norm_t = torch.from_numpy(y_norm).float()

    # to device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    edge_index = edge_index.to(device)
    y_norm_t = y_norm_t.to(device)
    train_mask_t = torch.from_numpy(train_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)

    in_dim = x.shape[1]
    out_dim = y.shape[1]

    # ---------------- MLP (UNCHANGED) ----------------
    mlp = MLP(in_dim, out_dim, hidden=args.hidden_mlp, dropout=args.dropout).to(device)
    opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_mlp = (1e18, None)  # (loss, state)
    patience = 0
    for ep in range(1, args.epochs + 1):
        loss = train_one(mlp, opt_mlp, x, edge_index, y_norm_t, train_mask_t, device, is_gnn=False)
        with torch.no_grad():
            pr = predict(mlp, x, edge_index, device, is_gnn=False)
            val_loss = F.mse_loss(pr[test_mask_t], y_norm_t[test_mask_t]).item()

        if val_loss < best_mlp[0] - 1e-5:
            best_mlp = (val_loss, {k: v.detach().cpu() for k, v in mlp.state_dict().items()})
            patience = 0
        else:
            patience += 1

        if ep % 50 == 0 or ep == 1:
            log(f"[MLP] ep={ep:04d} train_loss={loss:.4f} test_mse(z)={val_loss:.4f} patience={patience}", fplog)

        if patience >= args.early_patience:
            log(f"[MLP] early stop at ep={ep}, best_test_mse(z)={best_mlp[0]:.4f}", fplog)
            break

    mlp.load_state_dict(best_mlp[1])
    ckpt_mlp = os.path.join(args.out_dir, "ckpt", f"{args.patient}_mlp.pt")
    torch.save({"state_dict": best_mlp[1], "args": vars(args)}, ckpt_mlp)
    log(f"[MLP] saved: {ckpt_mlp}", fplog)

    # ---------------- GNN (UNCHANGED) ----------------
    gnn = GraphSAGE(in_dim, out_dim, hidden=args.hidden_gnn, dropout=args.dropout).to(device)
    opt_gnn = torch.optim.AdamW(gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_gnn = (1e18, None)
    patience = 0
    for ep in range(1, args.epochs + 1):
        loss = train_one(gnn, opt_gnn, x, edge_index, y_norm_t, train_mask_t, device, is_gnn=True)
        with torch.no_grad():
            pr = predict(gnn, x, edge_index, device, is_gnn=True)
            val_loss = F.mse_loss(pr[test_mask_t], y_norm_t[test_mask_t]).item()

        if val_loss < best_gnn[0] - 1e-5:
            best_gnn = (val_loss, {k: v.detach().cpu() for k, v in gnn.state_dict().items()})
            patience = 0
        else:
            patience += 1

        if ep % 50 == 0 or ep == 1:
            log(f"[GNN] ep={ep:04d} train_loss={loss:.4f} test_mse(z)={val_loss:.4f} patience={patience}", fplog)

        if patience >= args.early_patience:
            log(f"[GNN] early stop at ep={ep}, best_test_mse(z)={best_gnn[0]:.4f}", fplog)
            break

    gnn.load_state_dict(best_gnn[1])
    ckpt_gnn = os.path.join(args.out_dir, "ckpt", f"{args.patient}_gnn.pt")
    torch.save({"state_dict": best_gnn[1], "args": vars(args)}, ckpt_gnn)
    log(f"[GNN] saved: {ckpt_gnn}", fplog)

    # ---------------- DualGraph (NEW, does NOT affect MLP/GNN) ----------------
    dual = None
    best_dual = None
    if args.pathway_edges_tsv:
        log(f"[DUAL] pathway_edges_tsv: {args.pathway_edges_tsv}", fplog)
        A_norm = load_pathway_graph_A_norm(colnames, args.pathway_edges_tsv, device=device)

        dual = DualGraphSAGE(in_dim, out_dim, A_norm_sparse=A_norm,
                             hidden=args.hidden_gnn, dropout=args.dropout).to(device)
        opt_dual = torch.optim.AdamW(dual.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_dual = (1e18, None)
        patience = 0
        for ep in range(1, args.epochs + 1):
            loss = train_one(dual, opt_dual, x, edge_index, y_norm_t, train_mask_t, device, is_gnn=True)
            with torch.no_grad():
                pr = predict(dual, x, edge_index, device, is_gnn=True)
                val_loss = F.mse_loss(pr[test_mask_t], y_norm_t[test_mask_t]).item()

            if val_loss < best_dual[0] - 1e-5:
                best_dual = (val_loss, {k: v.detach().cpu() for k, v in dual.state_dict().items()})
                patience = 0
            else:
                patience += 1

            if ep % 50 == 0 or ep == 1:
                log(f"[DUAL] ep={ep:04d} train_loss={loss:.4f} test_mse(z)={val_loss:.4f} patience={patience}", fplog)

            if patience >= args.early_patience:
                log(f"[DUAL] early stop at ep={ep}, best_test_mse(z)={best_dual[0]:.4f}", fplog)
                break

        dual.load_state_dict(best_dual[1])
        ckpt_dual = os.path.join(args.out_dir, "ckpt", f"{args.patient}_dualgraph.pt")
        torch.save({"state_dict": best_dual[1], "args": vars(args)}, ckpt_dual)
        # also save alpha
        try:
            alpha = float(torch.sigmoid(dual.path_head.logit_alpha).detach().cpu().item())
            log(f"[DUAL] saved: {ckpt_dual} (alpha={alpha:.4f})", fplog)
        except Exception:
            log(f"[DUAL] saved: {ckpt_dual}", fplog)

    # ---------------- Save predictions (original scale) ----------------
    with torch.no_grad():
        pred_mlp_z = predict(mlp, x, edge_index, device, is_gnn=False).detach().cpu().numpy()
        pred_gnn_z = predict(gnn, x, edge_index, device, is_gnn=True).detach().cpu().numpy()
        pred_dual_z = None
        if dual is not None:
            pred_dual_z = predict(dual, x, edge_index, device, is_gnn=True).detach().cpu().numpy()

    gt = y  # original scale
    pred_mlp = denormalize_y(pred_mlp_z, norm)
    pred_gnn = denormalize_y(pred_gnn_z, norm)
    pred_dual = denormalize_y(pred_dual_z, norm) if pred_dual_z is not None else None

    np.save(os.path.join(args.out_dir, "preds", f"{args.patient}_gt.npy"), gt)
    np.save(os.path.join(args.out_dir, "preds", f"{args.patient}_pred_mlp.npy"), pred_mlp)
    np.save(os.path.join(args.out_dir, "preds", f"{args.patient}_pred_gnn.npy"), pred_gnn)
    if pred_dual is not None:
        np.save(os.path.join(args.out_dir, "preds", f"{args.patient}_pred_dualgraph.npy"), pred_dual)

    np.save(os.path.join(args.out_dir, "preds", f"{args.patient}_test_mask.npy"), test_mask.astype(np.uint8))
    np.save(os.path.join(args.out_dir, "preds", f"{args.patient}_pos.npy"), pos.astype(np.float32))
    with open(os.path.join(args.out_dir, "preds", f"{args.patient}_pathways.json"), "w", encoding="utf-8") as f:
        json.dump(colnames, f, ensure_ascii=False, indent=2)

    log("[save] preds/gt/test_mask/pos/pathways saved.", fplog)

    # ---------------- Metrics per pathway ----------------
    df_mlp = compute_metrics_per_pathway(gt, pred_mlp, test_mask, colnames)
    df_gnn = compute_metrics_per_pathway(gt, pred_gnn, test_mask, colnames)

    df = df_gnn.rename(columns={"pearson":"pearson_gnn","r2":"r2_gnn","rmse":"rmse_gnn",
                                "gt_mean":"gt_mean","gt_nz_ratio":"gt_nz_ratio"}) \
              .merge(df_mlp[["pathway","pearson","r2","rmse"]].rename(
                    columns={"pearson":"pearson_mlp","r2":"r2_mlp","rmse":"rmse_mlp"}),
                    on="pathway", how="left")

    df["delta_pearson"] = df["pearson_gnn"] - df["pearson_mlp"]
    df["delta_r2"] = df["r2_gnn"] - df["r2_mlp"]

    if pred_dual is not None:
        df_dual = compute_metrics_per_pathway(gt, pred_dual, test_mask, colnames)
        df = df.merge(
            df_dual[["pathway","pearson","r2","rmse"]].rename(
                columns={"pearson":"pearson_dual","r2":"r2_dual","rmse":"rmse_dual"}
            ),
            on="pathway", how="left"
        )
        df["delta_pearson_dual_vs_gnn"] = df["pearson_dual"] - df["pearson_gnn"]
        df["delta_r2_dual_vs_gnn"] = df["r2_dual"] - df["r2_gnn"]

    out_metrics = os.path.join(args.out_dir, "metrics", f"{args.patient}_pathway_metrics.csv")
    df.sort_values(["pearson_gnn"], ascending=False).to_csv(out_metrics, index=False)
    log(f"[metrics] saved: {out_metrics}", fplog)

    # quick summary
    cols_show = ["pathway","pearson_gnn","pearson_mlp","delta_pearson"]
    if pred_dual is not None:
        cols_show += ["pearson_dual","delta_pearson_dual_vs_gnn"]
    top = df.sort_values("pearson_gnn", ascending=False).head(10)[cols_show]

    log("\nTop-10 by pearson_gnn:", fplog)
    log(top.to_string(index=False), fplog)

    fplog.close()
    print(f"\nDONE. Log: {log_fp}")

if __name__ == "__main__":
    main()



# RUN_DIR=/home/xiaoxinyu/代谢/scFEA/output/ccRCC/within_patient_runs/Y27_pathway_blocksplit_v2_dualgraph
# mkdir -p $RUN_DIR

# python /home/xiaoxinyu/代谢/TEXT/within-test-pathway/02_within_patient_train_eval_dualgraph.py \
#   --graph_dir /home/xiaoxinyu/代谢/scFEA/output/ccRCC/pyg_graphs_v6b_intersect \
#   --y_dir /home/xiaoxinyu/代谢/scFEA/input/ccRCC/new_data/pathway_level_v2_filtered_spot_intersect \
#   --patient Y27 \
#   --out_dir $RUN_DIR \
#   --device cuda:4 \
#   --seed 0 \
#   --grid 6,6 \
#   --test_frac 0.8 \
#   --epochs 800 \
#   --early_patience 80 \
#   --pathway_edges_tsv /home/xiaoxinyu/代谢/scFEA/input/ccRCC/new_data/pathway_level_v2_filtered/pathway_edges.tsv
