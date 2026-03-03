#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Fixed colors (match your convention)
# ----------------------------
COLOR_MLP  = "#1f77b4"  # blue
COLOR_GNN  = "#2ca02c"  # green
COLOR_DUAL = "#ff7f0e"  # orange

# ----------------------------
# Annotation: read pathways_kept.txt
# ----------------------------
def read_pathway_annot(txt_path: str):
    """
    pathways_kept.txt line:
      map00010\tGlycolysis / Gluconeogenesis\tdeg=62\tcpd=8
    return dict: map_id -> dict(name, deg, cpd)
    """
    mp = {}
    if not txt_path:
        return mp
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pid = parts[0].strip()
            name = parts[1].strip()
            deg = None
            cpd = None
            for p in parts[2:]:
                if p.startswith("deg="):
                    try: deg = int(p.split("=",1)[1])
                    except: pass
                if p.startswith("cpd="):
                    try: cpd = int(p.split("=",1)[1])
                    except: pass
            mp[pid] = {"name": name, "deg": deg, "cpd": cpd}
    return mp

def annotate(df: pd.DataFrame, annot_map: dict) -> pd.DataFrame:
    if not annot_map:
        df["pathway_name"] = ""
        df["deg"] = np.nan
        df["cpd"] = np.nan
    else:
        df["pathway_name"] = df["pathway"].map(lambda x: annot_map.get(str(x), {}).get("name",""))
        df["deg"] = df["pathway"].map(lambda x: annot_map.get(str(x), {}).get("deg", np.nan))
        df["cpd"] = df["pathway"].map(lambda x: annot_map.get(str(x), {}).get("cpd", np.nan))

    df["label"] = df["pathway"] + df["pathway_name"].map(lambda s: f" | {s}" if s else "")
    return df

def savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def _need_cols(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"metrics missing columns: {miss}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=25)
    ap.add_argument("--annot_txt", default="", help="pathways_kept.txt (recommended)")
    ap.add_argument("--patient", default="", help="optional title prefix, e.g. Y27")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.metrics_csv)

    # ---- required columns (old baseline) ----
    base_need = [
        "pathway",
        "pearson_gnn","pearson_mlp","delta_pearson",
        "r2_gnn","r2_mlp",
        "rmse_gnn","rmse_mlp",
        "gt_nz_ratio"
    ]
    _need_cols(df, base_need)

    # ---- dualgraph optional columns ----
    has_dual = ("pearson_dual" in df.columns)
    # if has dual, we expect these too (but be tolerant)
    # delta_pearson_dual_vs_gnn is most important
    if has_dual:
        # be tolerant: if missing, we will compute where possible
        if "delta_pearson_dual_vs_gnn" not in df.columns:
            df["delta_pearson_dual_vs_gnn"] = df["pearson_dual"] - df["pearson_gnn"]
        if "delta_pearson_dual_vs_mlp" not in df.columns:
            df["delta_pearson_dual_vs_mlp"] = df["pearson_dual"] - df["pearson_mlp"]

    # annotate
    annot_map = read_pathway_annot(args.annot_txt)
    df = annotate(df, annot_map)

    title_prefix = (args.patient + " | ") if args.patient else ""

    # sort views
    df_g = df.sort_values("pearson_gnn", ascending=False).reset_index(drop=True)
    df_d = df.sort_values("delta_pearson", ascending=False).reset_index(drop=True)
    df_w = df.sort_values("pearson_gnn", ascending=True).reset_index(drop=True)

    # Dual ranking views (if exists)
    if has_dual:
        df_dual = df.sort_values("pearson_dual", ascending=False).reset_index(drop=True)
        df_d_dual = df.sort_values("delta_pearson_dual_vs_gnn", ascending=False).reset_index(drop=True)

    # =========================
    # (1) histogram: pearson distribution
    # =========================
    plt.figure(figsize=(7.5,4.5))
    plt.hist(df["pearson_mlp"].values, bins=25, alpha=0.55, label="MLP", color=COLOR_MLP)
    plt.hist(df["pearson_gnn"].values, bins=25, alpha=0.55, label="GNN", color=COLOR_GNN)
    if has_dual:
        plt.hist(df["pearson_dual"].values, bins=25, alpha=0.55, label="DualGraph", color=COLOR_DUAL)
    plt.xlabel("Pearson on test spots")
    plt.ylabel("Count (pathways)")
    plt.legend()
    plt.title(f"{title_prefix}Distribution of pathway predictability")
    savefig(os.path.join(out_dir, "01_hist_pearson.png"))

    # =========================
    # (2) scatter: gnn vs mlp (+ dual)
    # =========================
    plt.figure(figsize=(6,6))
    plt.scatter(df["pearson_mlp"], df["pearson_gnn"], s=20, alpha=0.65, color=COLOR_GNN, label="GNN vs MLP")
    lim = [
        min(df["pearson_mlp"].min(), df["pearson_gnn"].min()) - 0.03,
        max(df["pearson_mlp"].max(), df["pearson_gnn"].max()) + 0.03
    ]
    plt.plot(lim, lim, linewidth=1, color="gray")
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("Pearson (MLP)")
    plt.ylabel("Pearson (GNN)")
    plt.title(f"{title_prefix}GNN vs MLP (each dot = pathway)")
    plt.legend(loc="lower right")
    savefig(os.path.join(out_dir, "02_scatter_gnn_vs_mlp.png"))

    if has_dual:
        plt.figure(figsize=(6,6))
        plt.scatter(df["pearson_gnn"], df["pearson_dual"], s=20, alpha=0.65, color=COLOR_DUAL, label="Dual vs GNN")
        lim2 = [
            min(df["pearson_gnn"].min(), df["pearson_dual"].min()) - 0.03,
            max(df["pearson_gnn"].max(), df["pearson_dual"].max()) + 0.03
        ]
        plt.plot(lim2, lim2, linewidth=1, color="gray")
        plt.xlim(lim2); plt.ylim(lim2)
        plt.xlabel("Pearson (GNN)")
        plt.ylabel("Pearson (DualGraph)")
        plt.title(f"{title_prefix}DualGraph vs GNN (each dot = pathway)")
        plt.legend(loc="lower right")
        savefig(os.path.join(out_dir, "02b_scatter_dual_vs_gnn.png"))

    # =========================
    # (3) delta distributions
    # =========================
    plt.figure(figsize=(7.5,4.5))
    plt.hist(df["delta_pearson"].values, bins=25, alpha=0.8, color=COLOR_GNN)
    plt.axvline(0, linewidth=1, color="gray")
    plt.xlabel("delta_pearson = pearson_gnn - pearson_mlp")
    plt.ylabel("Count (pathways)")
    plt.title(f"{title_prefix}Spatial gain distribution (GNN-MLP)")
    savefig(os.path.join(out_dir, "03_hist_delta_gnn_minus_mlp.png"))

    if has_dual:
        plt.figure(figsize=(7.5,4.5))
        plt.hist(df["delta_pearson_dual_vs_gnn"].values, bins=25, alpha=0.8, color=COLOR_DUAL)
        plt.axvline(0, linewidth=1, color="gray")
        plt.xlabel("delta = pearson_dual - pearson_gnn")
        plt.ylabel("Count (pathways)")
        plt.title(f"{title_prefix}DualGraph gain distribution (Dual-GNN)")
        savefig(os.path.join(out_dir, "03b_hist_delta_dual_minus_gnn.png"))

    # =========================
    # (4) predictability vs sparsity
    # =========================
    plt.figure(figsize=(7,5))
    plt.scatter(df["gt_nz_ratio"], df["pearson_mlp"], s=18, alpha=0.5, color=COLOR_MLP, label="MLP")
    plt.scatter(df["gt_nz_ratio"], df["pearson_gnn"], s=18, alpha=0.55, color=COLOR_GNN, label="GNN")
    if has_dual:
        plt.scatter(df["gt_nz_ratio"], df["pearson_dual"], s=18, alpha=0.55, color=COLOR_DUAL, label="DualGraph")
    plt.xlabel("gt_nz_ratio (non-zero fraction)")
    plt.ylabel("Pearson")
    plt.title(f"{title_prefix}Predictability vs sparsity")
    plt.legend()
    savefig(os.path.join(out_dir, "04_scatter_pearson_vs_nzratio.png"))

    # =========================
    # (5) Top-K bar: pearson (GNN vs MLP vs Dual)
    # =========================
    top = df_g.head(args.top_k).copy()
    plt.figure(figsize=(12, max(4, 0.28*len(top))))
    y = np.arange(len(top))[::-1]
    # plot grouped bars (horizontal)
    h = 0.25
    yy = y.astype(float)
    plt.barh(yy + h, top["pearson_mlp"].values[::-1], height=h, color=COLOR_MLP, label="MLP")
    plt.barh(yy,     top["pearson_gnn"].values[::-1], height=h, color=COLOR_GNN, label="GNN")
    if has_dual:
        plt.barh(yy - h, top["pearson_dual"].values[::-1], height=h, color=COLOR_DUAL, label="DualGraph")
    plt.yticks(y, top["label"].values[::-1], fontsize=8)
    plt.xlabel("Pearson (test)")
    plt.title(f"{title_prefix}Top-{args.top_k} pathways (ranked by Pearson GNN)")
    plt.legend(loc="lower right")
    savefig(os.path.join(out_dir, "05_topk_pearson_compare.png"))

    # =========================
    # (6) Bottom-K bar: pearson compare
    # =========================
    bot = df_w.head(args.top_k).copy()
    plt.figure(figsize=(12, max(4, 0.28*len(bot))))
    y = np.arange(len(bot))[::-1]
    h = 0.25
    yy = y.astype(float)
    plt.barh(yy + h, bot["pearson_mlp"].values[::-1], height=h, color=COLOR_MLP, label="MLP")
    plt.barh(yy,     bot["pearson_gnn"].values[::-1], height=h, color=COLOR_GNN, label="GNN")
    if has_dual:
        plt.barh(yy - h, bot["pearson_dual"].values[::-1], height=h, color=COLOR_DUAL, label="DualGraph")
    plt.yticks(y, bot["label"].values[::-1], fontsize=8)
    plt.xlabel("Pearson (test)")
    plt.title(f"{title_prefix}Bottom-{args.top_k} pathways (ranked by Pearson GNN)")
    plt.legend(loc="lower right")
    savefig(os.path.join(out_dir, "06_bottomk_pearson_compare.png"))

    # =========================
    # (7) Top-K gains
    # =========================
    gain = df_d.head(args.top_k).copy()
    plt.figure(figsize=(12, max(4, 0.28*len(gain))))
    y = np.arange(len(gain))[::-1]
    plt.barh(y, gain["delta_pearson"].values[::-1], color=COLOR_GNN)
    plt.yticks(y, gain["label"].values[::-1], fontsize=8)
    plt.axvline(0, linewidth=1, color="gray")
    plt.xlabel("delta_pearson (GNN-MLP)")
    plt.title(f"{title_prefix}Top-{args.top_k} pathways by spatial gain (GNN-MLP)")
    savefig(os.path.join(out_dir, "07_topk_delta_gnn_minus_mlp.png"))

    if has_dual:
        gain2 = df_d_dual.head(args.top_k).copy()
        plt.figure(figsize=(12, max(4, 0.28*len(gain2))))
        y = np.arange(len(gain2))[::-1]
        plt.barh(y, gain2["delta_pearson_dual_vs_gnn"].values[::-1], color=COLOR_DUAL)
        plt.yticks(y, gain2["label"].values[::-1], fontsize=8)
        plt.axvline(0, linewidth=1, color="gray")
        plt.xlabel("delta (Dual-GNN)")
        plt.title(f"{title_prefix}Top-{args.top_k} pathways by DualGraph gain (Dual-GNN)")
        savefig(os.path.join(out_dir, "07b_topk_delta_dual_minus_gnn.png"))

    # =========================
    # (8) “volcano-ish”: gain vs predictability
    # =========================
    plt.figure(figsize=(6.5,5.5))
    plt.scatter(df["delta_pearson"], df["pearson_gnn"], s=18, alpha=0.7, color=COLOR_GNN)
    plt.axvline(0, linewidth=1, color="gray")
    plt.xlabel("delta_pearson (GNN-MLP)")
    plt.ylabel("pearson_gnn")
    plt.title(f"{title_prefix}Spatial gain vs predictability")
    savefig(os.path.join(out_dir, "08_scatter_delta_gnn_minus_mlp_vs_pearson_gnn.png"))

    if has_dual:
        plt.figure(figsize=(6.5,5.5))
        plt.scatter(df["delta_pearson_dual_vs_gnn"], df["pearson_dual"], s=18, alpha=0.7, color=COLOR_DUAL)
        plt.axvline(0, linewidth=1, color="gray")
        plt.xlabel("delta (Dual-GNN)")
        plt.ylabel("pearson_dual")
        plt.title(f"{title_prefix}Dual gain vs Dual predictability")
        savefig(os.path.join(out_dir, "08b_scatter_delta_dual_minus_gnn_vs_pearson_dual.png"))

    # =========================
    # (9) save ranked table
    # =========================
    out_rank = os.path.join(out_dir, "ranked_pathways.csv")
    df_g.to_csv(out_rank, index=False)

    # =========================
    # (10) summary json
    # =========================
    summ = {
        "n_pathways": int(len(df)),
        "pearson_mlp_mean": float(df["pearson_mlp"].mean()),
        "pearson_gnn_mean": float(df["pearson_gnn"].mean()),
        "pearson_gnn_median": float(df["pearson_gnn"].median()),
        "delta_gnn_minus_mlp_mean": float(df["delta_pearson"].mean()),
        "delta_gnn_minus_mlp_median": float(df["delta_pearson"].median()),
        "nz_ratio_median": float(df["gt_nz_ratio"].median()),
        "top10_pearson_gnn_mean": float(df_g.head(10)["pearson_gnn"].mean()),
    }
    if has_dual:
        summ.update({
            "pearson_dual_mean": float(df["pearson_dual"].mean()),
            "pearson_dual_median": float(df["pearson_dual"].median()),
            "delta_dual_minus_gnn_mean": float(df["delta_pearson_dual_vs_gnn"].mean()),
            "delta_dual_minus_gnn_median": float(df["delta_pearson_dual_vs_gnn"].median()),
            "top10_pearson_dual_mean": float(df_dual.head(10)["pearson_dual"].mean()),
        })

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)

    print("Saved figures to:", out_dir)
    print("Saved ranked table:", out_rank)
    print("Saved summary.json")
    print("DualGraph detected:", has_dual)

if __name__ == "__main__":
    main()


# python /home/xiaoxinyu/代谢/TEXT/within-test-pathway/04_plot_metrics_dashboard.py \
#   --metrics_csv /home/xiaoxinyu/代谢/scFEA/output/ccRCC/within_patient_runs/Y27_pathway_blocksplit_v2_dualgraph/metrics/Y27_pathway_metrics.csv \
#   --annot_txt /home/xiaoxinyu/代谢/scFEA/input/ccRCC/new_data/pathway_level_v2_filtered/pathways_kept.txt \
#   --out_dir /home/xiaoxinyu/代谢/scFEA/output/ccRCC/within_patient_runs/Y27_pathway_blocksplit_v2_dualgraph/metrics_extra_figs \
#   --patient Y27 \
#   --top_k 25
