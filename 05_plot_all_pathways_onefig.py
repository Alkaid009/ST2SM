#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_pathway_annot(txt_path: str):
    """
    pathways_kept.txt example line:
      map00010\tGlycolysis / Gluconeogenesis\tdeg=62\tcpd=8
    return dict: map_id -> dict(name, deg, cpd)
    """
    mp = {}
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
                    try:
                        deg = int(p.split("=", 1)[1])
                    except:
                        pass
                if p.startswith("cpd="):
                    try:
                        cpd = int(p.split("=", 1)[1])
                    except:
                        pass
            mp[pid] = {"name": name, "deg": deg, "cpd": cpd}
    return mp


def zscore_row(x: np.ndarray, eps=1e-8):
    x = x.astype(np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mu) / (sd + eps)


def _require_cols(df: pd.DataFrame, cols, name="metrics_csv"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--annot_txt", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--patient", default="Y27")
    ap.add_argument("--show_every", type=int, default=10, help="show every N pathway labels")
    ap.add_argument("--sort_by", default="pearson_gnn",
                    help="sort key: pearson_gnn (default) or pearson_dual if present etc.")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_csv)

    # base columns must exist (from your new training script)
    base_need = [
        "pathway",
        "pearson_gnn", "pearson_mlp", "delta_pearson",
        "r2_gnn", "r2_mlp", "rmse_gnn", "rmse_mlp",
        "gt_nz_ratio",
    ]
    _require_cols(df, base_need)

    # detect dualgraph columns
    has_dual = all(c in df.columns for c in [
        "pearson_dual", "r2_dual", "rmse_dual", "delta_pearson_dual_vs_gnn"
    ])

    annot = read_pathway_annot(args.annot_txt)
    df["pathway_name"] = df["pathway"].map(lambda x: annot.get(str(x), {}).get("name", ""))
    df["deg"] = df["pathway"].map(lambda x: annot.get(str(x), {}).get("deg", np.nan))
    df["cpd"] = df["pathway"].map(lambda x: annot.get(str(x), {}).get("cpd", np.nan))

    # sort
    sort_key = args.sort_by
    if sort_key not in df.columns:
        sort_key = "pearson_dual" if has_dual else "pearson_gnn"
    df = df.sort_values(sort_key, ascending=False).reset_index(drop=True)

    pathways = df["pathway"].astype(str).tolist()
    x = np.arange(len(df))

    # heat rows: row-wise zscore (so rows comparable visually)
    heat_rows = [
        ("r2_gnn",   df["r2_gnn"].values),
        ("r2_mlp",   df["r2_mlp"].values),
    ]
    if has_dual:
        heat_rows.append(("r2_dual", df["r2_dual"].values))

    heat_rows += [
        ("rmse_gnn", df["rmse_gnn"].values),
        ("rmse_mlp", df["rmse_mlp"].values),
    ]
    if has_dual:
        heat_rows.append(("rmse_dual", df["rmse_dual"].values))

    heat_rows += [
        ("nz_ratio", df["gt_nz_ratio"].values),
    ]

    H = np.vstack([zscore_row(v) for _, v in heat_rows])

    # ---------------------------------------------------------
    # Plot (ONE figure, 3 panels)
    # ---------------------------------------------------------
    plt.figure(figsize=(20, 11))

    # (A) pearson curves
    ax1 = plt.subplot2grid((12, 1), (0, 0), rowspan=5)
    ax1.plot(x, df["pearson_gnn"].values, label="pearson_gnn")
    ax1.plot(x, df["pearson_mlp"].values, label="pearson_mlp")
    if has_dual:
        ax1.plot(x, df["pearson_dual"].values, label="pearson_dual")
    ax1.set_ylabel("Pearson (test)")
    title = f"{args.patient} | All pathways ranked by {sort_key} (n={len(df)})"
    if has_dual:
        title += " | DualGraph ON"
    else:
        title += " | DualGraph OFF"
    ax1.set_title(title)
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.2)

    # mark top/bottom zones lightly
    ax1.axvspan(0, min(20, len(df)-1), alpha=0.05)
    ax1.axvspan(max(0, len(df)-20), len(df)-1, alpha=0.05)

    # (B) delta pearson bars
    ax2 = plt.subplot2grid((12, 1), (5, 0), rowspan=3, sharex=ax1)
    ax2.bar(x, df["delta_pearson"].values, label="ΔPearson (GNN-MLP)")
    if has_dual:
        ax2.bar(x, df["delta_pearson_dual_vs_gnn"].values,
                alpha=0.6, label="ΔPearson (Dual-GNN)")
    ax2.axhline(0, linewidth=1)
    ax2.set_ylabel("ΔPearson")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper right")

    # (C) heatmap
    ax3 = plt.subplot2grid((12, 1), (8, 0), rowspan=4, sharex=ax1)
    im = ax3.imshow(H, aspect="auto", interpolation="nearest")
    ax3.set_yticks(np.arange(len(heat_rows)))
    ax3.set_yticklabels([k for k, _ in heat_rows])
    ax3.set_xlabel("Pathways (ranked)")
    cbar = plt.colorbar(im, ax=ax3, fraction=0.02, pad=0.01)
    cbar.set_label("Row z-score")

    # x tick labels (sparse)
    show_every = max(1, int(args.show_every))
    xt = np.arange(0, len(df), show_every)
    ax3.set_xticks(xt)
    ax3.set_xticklabels([pathways[i] for i in xt], rotation=90, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.savefig(args.out_png, dpi=220)
    plt.close()

    print("Saved:", args.out_png)

    # also save a ranked table with names for convenience
    out_csv = os.path.splitext(args.out_png)[0] + "_ranked_with_names.csv"
    df.to_csv(out_csv, index=False)
    print("Saved table:", out_csv)


if __name__ == "__main__":
    main()

# Example:
# python /home/xiaoxinyu/代谢/TEXT/within-test-pathway/05_plot_all_pathways_onefig.py \
#   --metrics_csv /home/xiaoxinyu/代谢/scFEA/output/ccRCC/within_patient_runs/Y27_pathway_blocksplit_v2_dualgraph/metrics/Y27_pathway_metrics.csv \
#   --annot_txt /home/xiaoxinyu/代谢/scFEA/input/ccRCC/new_data/pathway_level_v2_filtered/pathways_kept.txt \
#   --out_png /home/xiaoxinyu/代谢/scFEA/output/ccRCC/within_patient_runs/Y27_pathway_blocksplit_v2_dualgraph/metrics_figs/Y27_all_pathways_onefig.png \
#   --patient Y27 \
#   --show_every 10 \
#   --sort_by pearson_gnn
