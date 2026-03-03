#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Colors (fixed)
# ----------------------------
COLOR_MLP  = "#1f77b4"  # blue
COLOR_GNN  = "#2ca02c"  # green
COLOR_DUAL = "#ff7f0e"  # orange


# ----------------------------
# 1) Keyword-based categories (editable)
# Order matters: first hit wins.
# ----------------------------
CATEGORY_RULES = [
    ("Carbohydrate metabolism", [
        "Glycolysis", "Gluconeogenesis", "Pentose", "Fructose", "Mannose",
        "Galactose", "Starch", "Sucrose", "Pyruvate", "Citrate", "TCA",
        "Glyoxylate", "Butanoate", "Propanoate", "Inositol", "Glucuronate",
        "Ascorbate", "Aldarate"
    ]),
    ("Lipid metabolism", [
        "Fatty acid", "Glycerolipid", "Glycerophospholipid", "Sphingolipid",
        "Steroid", "Cholesterol", "Bile", "Arachidonic", "Linoleic", "Alpha-linolenic",
        "Ether lipid", "Terpenoid", "Carotenoid", "Porphyrin", "Retinol"
    ]),
    ("Amino acid metabolism", [
        "Alanine", "Aspartate", "Glutamate", "Glycine", "Serine", "Threonine",
        "Cysteine", "Methionine", "Valine", "Leucine", "Isoleucine",
        "Lysine", "Arginine", "Proline", "Histidine", "Phenylalanine",
        "Tyrosine", "Tryptophan", "Branched-chain", "Urea", "Taurine"
    ]),
    ("Nucleotide metabolism", [
        "Purine", "Pyrimidine", "Riboflavin", "Nicotinate", "Nicotinamide",
        "One carbon", "Folate"
    ]),
    ("Redox & detox", [
        "Glutathione", "Cytochrome", "Xenobiotics", "Drug metabolism",
        "Metabolism of xenobiotics", "Chemical carcinogenesis"
    ]),
    ("Glycan metabolism", [
        "Glycosaminoglycan", "Glycan", "N-Glycan", "O-Glycan", "Glycosylphosphatidylinositol",
        "GPI-anchor"
    ]),
    ("Cofactors & vitamins", [
        "Vitamin", "Biotin", "Thiamine", "Pantothenate", "CoA", "Cobalamin",
        "Pyridoxal", "Vitamin B", "Vitamin C", "Vitamin E"
    ]),
    ("Signal transduction / endocrine", [
        "signaling", "Signaling", "MAPK", "PI3K", "Akt", "mTOR", "HIF-",
        "VEGF", "TGF", "Wnt", "Notch", "Hippo", "cAMP", "cGMP",
        "Insulin", "Glucagon", "Thyroid", "Estrogen", "Androgen"
    ]),
    ("Immune / inflammation", [
        "Cytokine", "Chemokine", "Toll-like", "NOD-like", "NF-kappa",
        "Interleukin", "Complement", "Antigen", "Leukocyte"
    ]),
    ("Cell cycle / apoptosis / cancer", [
        "Cell cycle", "Apoptosis", "p53", "DNA replication", "Mismatch repair",
        "Base excision repair", "Homologous recombination", "Pathways in cancer",
        "Proteasome", "Ferroptosis"
    ]),
    ("Microbial / infectious", [
        "Bacterial", "Viral", "Infection", "pathogen", "Staphylococcus", "E. coli"
    ]),
    ("Secondary metabolite biosynthesis", [
        "biosynthesis", "Biosynthesis", "Staurosporine", "Alkaloid", "Flavonoid",
        "Polyketide", "Antibiotic"
    ]),
]

def read_pathways_kept(txt_path: str) -> pd.DataFrame:
    """
    pathways_kept.txt line:
      map00010\tGlycolysis / Gluconeogenesis\tdeg=62\tcpd=8
    """
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            pid = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else ""
            deg = np.nan
            cpd = np.nan
            for p in parts[2:]:
                if p.startswith("deg="):
                    try: deg = int(p.split("=",1)[1])
                    except: pass
                if p.startswith("cpd="):
                    try: cpd = int(p.split("=",1)[1])
                    except: pass
            rows.append((pid, name, deg, cpd))
    return pd.DataFrame(rows, columns=["pathway","pathway_name","deg","cpd"])

def assign_category(name: str) -> str:
    for cat, keywords in CATEGORY_RULES:
        for kw in keywords:
            if kw in name:
                return cat
    return "Other / unknown"

def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"metrics_csv missing columns: {missing}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--pathways_kept_txt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--patient", default="Y27")
    ap.add_argument("--min_group_size", type=int, default=5,
                    help="small groups will be merged into 'Other / small'")
    ap.add_argument("--scatter_use", choices=["auto", "gnn", "dual"], default="auto",
                    help="Panel C uses which pearson for y-axis. auto=dual if present else gnn.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    metrics = pd.read_csv(args.metrics_csv)
    ann = read_pathways_kept(args.pathways_kept_txt)

    # base columns (old script compatibility)
    base_need = ["pathway", "pearson_gnn", "pearson_mlp", "delta_pearson",
                 "rmse_gnn", "rmse_mlp", "r2_gnn", "r2_mlp", "gt_nz_ratio"]
    _require_cols(metrics, base_need)

    # detect dual columns
    has_dual = all(c in metrics.columns for c in [
        "pearson_dual", "r2_dual", "rmse_dual", "delta_pearson_dual_vs_gnn"
    ])

    df = metrics.merge(ann, on="pathway", how="left")
    df["pathway_name"] = df["pathway_name"].fillna("")
    df["category_raw"] = df["pathway_name"].map(assign_category)

    # merge small groups
    counts = df["category_raw"].value_counts()
    small = set(counts[counts < args.min_group_size].index.tolist())
    df["category"] = df["category_raw"].map(lambda c: "Other / small" if c in small else c)

    # add extra deltas if dual exists
    df = df.copy()
    df["delta_pearson_gnn_vs_mlp"] = df["pearson_gnn"] - df["pearson_mlp"]
    if has_dual:
        df["delta_pearson_dual_vs_gnn"] = df["pearson_dual"] - df["pearson_gnn"]
        df["delta_pearson_dual_vs_mlp"] = df["pearson_dual"] - df["pearson_mlp"]

    # sort categories by median pearson (dual preferred)
    sort_col_for_cat = "pearson_dual" if has_dual else "pearson_gnn"
    cat_order = (
        df.groupby("category")[sort_col_for_cat]
        .median().sort_values(ascending=False).index.tolist()
    )

    # ---------- outputs ----------
    out_table = os.path.join(args.out_dir, f"{args.patient}_pathway_category_table.csv")
    df.sort_values(sort_col_for_cat, ascending=False).to_csv(out_table, index=False)

    # summary per category
    agg_dict = dict(
        n=("pathway","count"),
        pearson_gnn_median=("pearson_gnn","median"),
        pearson_gnn_mean=("pearson_gnn","mean"),
        pearson_mlp_median=("pearson_mlp","median"),
        delta_gnn_vs_mlp_median=("delta_pearson_gnn_vs_mlp","median"),
        nz_ratio_median=("gt_nz_ratio","median"),
        rmse_gnn_median=("rmse_gnn","median"),
        rmse_mlp_median=("rmse_mlp","median"),
        r2_gnn_median=("r2_gnn","median"),
        r2_mlp_median=("r2_mlp","median"),
    )
    if has_dual:
        agg_dict.update(dict(
            pearson_dual_median=("pearson_dual","median"),
            pearson_dual_mean=("pearson_dual","mean"),
            delta_dual_vs_gnn_median=("delta_pearson_dual_vs_gnn","median"),
            delta_dual_vs_mlp_median=("delta_pearson_dual_vs_mlp","median"),
            rmse_dual_median=("rmse_dual","median"),
            r2_dual_median=("r2_dual","median"),
        ))

    agg = (
        df.groupby("category").agg(**agg_dict)
        .reset_index()
        .sort_values(("pearson_dual_median" if has_dual else "pearson_gnn_median"),
                     ascending=False)
    )

    out_sum = os.path.join(args.out_dir, f"{args.patient}_category_summary.csv")
    agg.to_csv(out_sum, index=False)

    # ---------- one-figure visualization ----------
    # Panel A: pearson_* by category (MLP/GNN/(Dual))
    # Panel B: delta pearson by category (GNN-MLP, (Dual-GNN))
    # Panel C: scatter nz_ratio vs pearson (gnn or dual)
    fig = plt.figure(figsize=(18, 11))

    # A: pearson
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    data_mlp = [df[df["category"]==c]["pearson_mlp"].values for c in cat_order]
    data_gnn = [df[df["category"]==c]["pearson_gnn"].values for c in cat_order]

    positions = np.arange(len(cat_order))
    width = 0.25 if has_dual else 0.35

    bp1 = ax1.boxplot(
        data_mlp, positions=positions - width, widths=width, patch_artist=True,
        showfliers=False
    )
    bp2 = ax1.boxplot(
        data_gnn, positions=positions, widths=width, patch_artist=True,
        showfliers=False
    )
    if has_dual:
        data_dual = [df[df["category"]==c]["pearson_dual"].values for c in cat_order]
        bp3 = ax1.boxplot(
            data_dual, positions=positions + width, widths=width, patch_artist=True,
            showfliers=False
        )

    ax1.set_title(f"{args.patient} | Pearson by category")
    ax1.set_ylabel("pearson (test)")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(cat_order, rotation=45, ha="right")

    # legend using dummy handles (avoid relying on box colors)
    handles = [
        plt.Line2D([0],[0], marker='s', linestyle='', label='MLP',
                markerfacecolor=COLOR_MLP, markeredgecolor=COLOR_MLP, markersize=10),
        plt.Line2D([0],[0], marker='s', linestyle='', label='GNN',
                markerfacecolor=COLOR_GNN, markeredgecolor=COLOR_GNN, markersize=10),
    ]
    if has_dual:
        handles.append(
            plt.Line2D([0],[0], marker='s', linestyle='', label='DualGraph',
                    markerfacecolor=COLOR_DUAL, markeredgecolor=COLOR_DUAL, markersize=10)
        )
    ax1.legend(handles=handles, loc="upper right", frameon=True)


    def _color_boxplot(bp, color):
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_edgecolor(color)
            box.set_alpha(0.35)
        for whisker in bp["whiskers"]:
            whisker.set_color(color)
        for cap in bp["caps"]:
            cap.set_color(color)
        for median in bp["medians"]:
            median.set_color(color)
            median.set_linewidth(2)

    _color_boxplot(bp1, COLOR_MLP)
    _color_boxplot(bp2, COLOR_GNN)
    if has_dual:
        _color_boxplot(bp3, COLOR_DUAL)

    # B: delta pearson
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    data_d1 = [df[df["category"]==c]["delta_pearson_gnn_vs_mlp"].values for c in cat_order]
    bp21 = ax2.boxplot(
        data_d1, positions=positions - (0.15 if has_dual else 0.0),
        widths=(0.3 if has_dual else 0.4), patch_artist=True, showfliers=False
    )
    if has_dual:
        data_d2 = [df[df["category"]==c]["delta_pearson_dual_vs_gnn"].values for c in cat_order]
        bp22 = ax2.boxplot(
            data_d2, positions=positions + 0.15,
            widths=0.3, patch_artist=True, showfliers=False
        )
    _color_boxplot(bp21, COLOR_GNN)  # GNN-MLP 用绿色
    if has_dual:
        _color_boxplot(bp22, COLOR_DUAL)  # Dual-GNN 用橙色

    ax2.axhline(0, linewidth=1)
    ax2.set_title(f"{args.patient} | ΔPearson by category")
    ax2.set_ylabel("Δpearson")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(cat_order, rotation=45, ha="right")
    handles2 = [
        plt.Line2D([0],[0], marker='s', linestyle='', label='GNN-MLP',
                markerfacecolor=COLOR_GNN, markeredgecolor=COLOR_GNN, markersize=10),
    ]
    if has_dual:
        handles2.append(
            plt.Line2D([0],[0], marker='s', linestyle='', label='Dual-GNN',
                    markerfacecolor=COLOR_DUAL, markeredgecolor=COLOR_DUAL, markersize=10)
        )
    ax2.legend(handles=handles2, loc="upper right", frameon=True)

    # C: scatter
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    if args.scatter_use == "gnn":
        ycol = "pearson_gnn"
        ylab = "pearson_gnn"
    elif args.scatter_use == "dual":
        ycol = "pearson_dual" if has_dual else "pearson_gnn"
        ylab = "pearson_dual" if has_dual else "pearson_gnn"
    else:
        # auto
        ycol = "pearson_dual" if has_dual else "pearson_gnn"
        ylab = ycol

    cat_to_i = {c:i for i,c in enumerate(cat_order)}
    cvals = df["category"].map(lambda c: cat_to_i.get(c, -1)).values
    ax3.scatter(df["gt_nz_ratio"].values, df[ycol].values, s=25, alpha=0.8, c=cvals)
    ax3.set_xlabel("gt_nz_ratio")
    ax3.set_ylabel(ylab)
    ax3.set_title(f"Each dot = pathway (colored by category) | y={ylab}")

    # legend (manual, keep readable)
    handles3 = []
    for c in cat_order:
        handles3.append(plt.Line2D([0],[0], marker='o', linestyle='',
                                   label=c, markersize=6))
    ax3.legend(handles=handles3, loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, f"{args.patient}_category_compare.png")
    plt.savefig(out_png, dpi=220)
    plt.close()

    print("Saved:")
    print("  table  :", out_table)
    print("  summary:", out_sum)
    print("  figure :", out_png)
    print("  dual   :", has_dual)


if __name__ == "__main__":
    main()

# Example:
# python /home/xiaoxinyu/代谢/TEXT/within-test-pathway/06_category_pathways.py \
#   --metrics_csv /home/xiaoxinyu/代谢/scFEA/output/ccRCC/within_patient_runs/Y27_pathway_blocksplit_v2_dualgraph/metrics/Y27_pathway_metrics.csv \
#   --pathways_kept_txt /home/xiaoxinyu/代谢/scFEA/input/ccRCC/new_data/pathway_level_v2_filtered/pathways_kept.txt \
#   --out_dir /home/xiaoxinyu/代谢/scFEA/output/ccRCC/within_patient_runs/Y27_pathway_blocksplit_v2_dualgraph/category_figs \
#   --patient Y27 \
#   --min_group_size 5 \
#   --scatter_use auto
