#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def load_tissue_positions(spatial_dir: str) -> pd.DataFrame:
    cand1 = os.path.join(spatial_dir, "tissue_positions_list.csv")
    cand2 = os.path.join(spatial_dir, "tissue_positions.csv")
    if os.path.exists(cand2):
        df = pd.read_csv(cand2)
        return df[["barcode","in_tissue","pxl_row_in_fullres","pxl_col_in_fullres"]].copy()
    if os.path.exists(cand1):
        df = pd.read_csv(cand1, header=None)
        df.columns = ["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]
        return df[["barcode","in_tissue","pxl_row_in_fullres","pxl_col_in_fullres"]].copy()
    raise FileNotFoundError(f"Cannot find tissue_positions*.csv under {spatial_dir}")


def load_scalefactors(spatial_dir: str) -> dict:
    fp = os.path.join(spatial_dir, "scalefactors_json.json")
    with open(fp, "r") as f:
        return json.load(f)


def load_image(spatial_dir: str, which: str):
    if which == "hires":
        cand = [os.path.join(spatial_dir, "tissue_hires_image.png"),
                os.path.join(spatial_dir, "tissue_hires_image.jpg")]
    else:
        cand = [os.path.join(spatial_dir, "tissue_lowres_image.png"),
                os.path.join(spatial_dir, "tissue_lowres_image.jpg")]
    for p in cand:
        if os.path.exists(p):
            return Image.open(p).convert("RGB"), p
    raise FileNotFoundError(f"Cannot find {which} tissue image under {spatial_dir}")


def overlay_one(ax, img, x, y, val, title, cmap="viridis",
                s=14, alpha=0.9, vmin=None, vmax=None):
    ax.imshow(img, origin="upper")
    sc = ax.scatter(x, y, c=val, s=s, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_title(title)
    ax.set_axis_off()
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--pathway", required=True, help="e.g. map01062")
    ap.add_argument("--y_csv", required=True, help="the intersect y csv used for training (index=barcode)")
    ap.add_argument("--spatial_dir", required=True, help="SpaceRanger output spatial/ dir")
    ap.add_argument("--use", choices=["hires","lowres"], default="hires")
    ap.add_argument("--only_in_tissue", action="store_true",
                    help="filter by in_tissue==1 from spaceranger (recommended)")
    ap.add_argument("--point_size", type=float, default=14)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--resid_cmap", default="coolwarm")
    args = ap.parse_args()

    run_dir = args.run_dir
    pid = args.patient
    pw = args.pathway

    # ---- load preds ----
    gt_fp = os.path.join(run_dir, "preds", f"{pid}_gt.npy")
    gnn_fp = os.path.join(run_dir, "preds", f"{pid}_pred_gnn.npy")
    mlp_fp = os.path.join(run_dir, "preds", f"{pid}_pred_mlp.npy")
    dual_fp = os.path.join(run_dir, "preds", f"{pid}_pred_dualgraph.npy")  # may not exist

    if not os.path.exists(gt_fp): raise FileNotFoundError(gt_fp)
    if not os.path.exists(gnn_fp): raise FileNotFoundError(gnn_fp)
    if not os.path.exists(mlp_fp): raise FileNotFoundError(mlp_fp)

    gt = np.load(gt_fp)
    pred_gnn = np.load(gnn_fp)
    pred_mlp = np.load(mlp_fp)

    has_dual = os.path.exists(dual_fp)
    pred_dual = np.load(dual_fp) if has_dual else None

    with open(os.path.join(run_dir, "preds", f"{pid}_pathways.json")) as f:
        pathways = json.load(f)
    if pw not in pathways:
        raise ValueError(f"pathway {pw} not found in {pid}_pathways.json")
    j = pathways.index(pw)

    # ---- alignment with y_csv ----
    y_df = pd.read_csv(args.y_csv, index_col=0)
    barcodes = list(y_df.index.astype(str))
    assert gt.shape[0] == len(barcodes), "gt not aligned with y_csv!"
    assert pred_gnn.shape[0] == len(barcodes), "pred_gnn not aligned with y_csv!"
    assert pred_mlp.shape[0] == len(barcodes), "pred_mlp not aligned with y_csv!"
    if has_dual:
        assert pred_dual.shape[0] == len(barcodes), "pred_dualgraph not aligned with y_csv!"

    # ---- load spatial image + coords ----
    tp = load_tissue_positions(args.spatial_dir)
    sf = load_scalefactors(args.spatial_dir)
    img, img_path = load_image(args.spatial_dir, args.use)
    scale = sf["tissue_hires_scalef"] if args.use == "hires" else sf["tissue_lowres_scalef"]

    # join to get pixel coords in chosen image scale
    df = pd.DataFrame({"barcode": barcodes, "i": np.arange(len(barcodes))})
    df = df.merge(tp, on="barcode", how="left")

    miss = int(df["pxl_row_in_fullres"].isna().sum())
    if miss > 0:
        print(f"[WARN] {miss} barcodes not found in tissue_positions. Example:",
              df[df["pxl_row_in_fullres"].isna()]["barcode"].head(5).tolist())

    if args.only_in_tissue:
        df = df[df["in_tissue"] == 1].copy()

    idx = df["i"].values.astype(int)

    # IMPORTANT: convert fullres pixel to hires/lowres pixel
    x = (df["pxl_col_in_fullres"].values * scale).astype(np.float32)
    y = (df["pxl_row_in_fullres"].values * scale).astype(np.float32)

    # ---- extract pathway values ----
    gt_pw = gt[idx, j]
    mlp_pw = pred_mlp[idx, j]
    gnn_pw = pred_gnn[idx, j]
    dual_pw = pred_dual[idx, j] if has_dual else None

    # consistent color range for GT & all predictions
    vals_for_range = [gt_pw, mlp_pw, gnn_pw]
    if has_dual:
        vals_for_range.append(dual_pw)
    vmin = float(min([v.min() for v in vals_for_range]))
    vmax = float(max([v.max() for v in vals_for_range]))

    # residuals
    resid_gnn = gnn_pw - gt_pw
    resid_mlp = mlp_pw - gt_pw
    resid_dual = (dual_pw - gt_pw) if has_dual else None

    out_dir = os.path.join(run_dir, "fig_overlay")
    os.makedirs(out_dir, exist_ok=True)
    out_fp = os.path.join(out_dir, f"{pid}_{pw}_{args.use}_overlay_all.png")

    # ---- layout ----
    # if dual exists: 2 rows × 3 cols = 6 panels
    # else: 1 row × 4 cols = 4 panels
    if has_dual:
        plt.figure(figsize=(22, 12))

        ax1 = plt.subplot(2, 3, 1)
        overlay_one(ax1, img, x, y, gt_pw,  f"{pid} {pw} GT ({args.use})",
                    cmap=args.cmap, s=args.point_size, alpha=args.alpha, vmin=vmin, vmax=vmax)

        ax2 = plt.subplot(2, 3, 2)
        overlay_one(ax2, img, x, y, mlp_pw, f"{pid} {pw} Pred(MLP) ({args.use})",
                    cmap=args.cmap, s=args.point_size, alpha=args.alpha, vmin=vmin, vmax=vmax)

        ax3 = plt.subplot(2, 3, 3)
        overlay_one(ax3, img, x, y, gnn_pw, f"{pid} {pw} Pred(GNN) ({args.use})",
                    cmap=args.cmap, s=args.point_size, alpha=args.alpha, vmin=vmin, vmax=vmax)

        ax4 = plt.subplot(2, 3, 4)
        overlay_one(ax4, img, x, y, dual_pw, f"{pid} {pw} Pred(DualGraph) ({args.use})",
                    cmap=args.cmap, s=args.point_size, alpha=args.alpha, vmin=vmin, vmax=vmax)

        ax5 = plt.subplot(2, 3, 5)
        overlay_one(ax5, img, x, y, resid_gnn, f"{pid} {pw} Resid(GNN-GT) ({args.use})",
                    cmap=args.resid_cmap, s=args.point_size, alpha=args.alpha)

        ax6 = plt.subplot(2, 3, 6)
        overlay_one(ax6, img, x, y, resid_dual, f"{pid} {pw} Resid(Dual-GT) ({args.use})",
                    cmap=args.resid_cmap, s=args.point_size, alpha=args.alpha)

    else:
        plt.figure(figsize=(24, 6))

        ax1 = plt.subplot(1, 4, 1)
        overlay_one(ax1, img, x, y, gt_pw,  f"{pid} {pw} GT ({args.use})",
                    cmap=args.cmap, s=args.point_size, alpha=args.alpha, vmin=vmin, vmax=vmax)

        ax2 = plt.subplot(1, 4, 2)
        overlay_one(ax2, img, x, y, mlp_pw, f"{pid} {pw} Pred(MLP) ({args.use})",
                    cmap=args.cmap, s=args.point_size, alpha=args.alpha, vmin=vmin, vmax=vmax)

        ax3 = plt.subplot(1, 4, 3)
        overlay_one(ax3, img, x, y, gnn_pw, f"{pid} {pw} Pred(GNN) ({args.use})",
                    cmap=args.cmap, s=args.point_size, alpha=args.alpha, vmin=vmin, vmax=vmax)

        ax4 = plt.subplot(1, 4, 4)
        overlay_one(ax4, img, x, y, resid_gnn, f"{pid} {pw} Resid(GNN-GT) ({args.use})",
                    cmap=args.resid_cmap, s=args.point_size, alpha=args.alpha)

    plt.tight_layout()
    plt.savefig(out_fp, dpi=220)
    plt.close()

    print("Saved:", out_fp)
    print("Used image:", img_path)
    if has_dual:
        print("Also used:", dual_fp)


if __name__ == "__main__":
    main()


# python /home/xiaoxinyu/代谢/TEXT/within-test-pathway/03_overlay_on_he.py \
#   --run_dir /home/xiaoxinyu/代谢/scFEA/output/ccRCC/within_patient_runs/Y27_pathway_blocksplit_v2_dualgraph \
#   --patient Y27 \
#   --pathway map04977 \
#   --y_csv /home/xiaoxinyu/代谢/scFEA/input/ccRCC/new_data/pathway_level_v2_filtered_spot_intersect/Y27_T_SM_Pathway.filtered.spot.intersect.csv \
#   --spatial_dir /home/xiaoxinyu/代谢/scFEA/input/ccRCC/ST_visium/Y27_T/outs/spatial \
#   --use hires \
#   --only_in_tissue
