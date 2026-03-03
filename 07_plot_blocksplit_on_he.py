#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def find_positions_csv(spatial_dir: str) -> str:
    cand = [
        os.path.join(spatial_dir, "tissue_positions.csv"),
        os.path.join(spatial_dir, "tissue_positions_list.csv"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"cannot find tissue_positions*.csv in {spatial_dir}")

def load_positions(spatial_dir: str) -> pd.DataFrame:
    pos_csv = find_positions_csv(spatial_dir)
    # tissue_positions_list.csv often has no header (10x old format)
    # Try header read; if fails, fallback.
    try:
        df = pd.read_csv(pos_csv)
        if "barcode" not in df.columns:
            raise ValueError("no barcode column")
    except Exception:
        df = pd.read_csv(
            pos_csv, header=None,
            names=["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]
        )
    # enforce columns
    need = ["barcode","in_tissue","pxl_row_in_fullres","pxl_col_in_fullres"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{pos_csv} missing columns: {miss}")
    df["barcode"] = df["barcode"].astype(str)
    df["in_tissue"] = df["in_tissue"].astype(int)
    return df

def load_scalefactors(spatial_dir: str) -> dict:
    sf = os.path.join(spatial_dir, "scalefactors_json.json")
    if not os.path.exists(sf):
        raise FileNotFoundError(sf)
    with open(sf, "r", encoding="utf-8") as f:
        return json.load(f)

def choose_image(spatial_dir: str, use: str):
    if use == "hires":
        img_path = os.path.join(spatial_dir, "tissue_hires_image.png")
        key = "tissue_hires_scalef"
    else:
        img_path = os.path.join(spatial_dir, "tissue_lowres_image.png")
        key = "tissue_lowres_scalef"
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    return img_path, key

def make_blocks(xy: np.ndarray, grid: tuple[int,int]):
    gx, gy = grid
    x = xy[:,0]; y = xy[:,1]
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    # avoid zero range
    if abs(x1-x0) < 1e-6: x1 = x0 + 1.0
    if abs(y1-y0) < 1e-6: y1 = y0 + 1.0

    dx = (x1 - x0) / gx
    dy = (y1 - y0) / gy

    # block index for each spot
    ix = np.clip(((x - x0) / dx).astype(int), 0, gx-1)
    iy = np.clip(((y - y0) / dy).astype(int), 0, gy-1)
    bid = iy * gx + ix

    # rectangles (in image coords)
    rects = []
    for j in range(gy):
        for i in range(gx):
            rx0 = x0 + i*dx
            ry0 = y0 + j*dy
            rects.append((rx0, ry0, dx, dy))
    return bid, rects, (x0, x1, y0, y1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial_dir", required=True, help=".../outs/spatial")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--grid", default="6,6", help="gx,gy")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use", choices=["hires","lowres"], default="hires")
    ap.add_argument("--only_in_tissue", action="store_true")
    ap.add_argument("--dot_size", type=float, default=6.0)
    args = ap.parse_args()

    gx, gy = [int(x) for x in args.grid.split(",")]
    assert gx > 0 and gy > 0

    df = load_positions(args.spatial_dir)
    if args.only_in_tissue:
        df = df[df["in_tissue"] == 1].copy()

    sf = load_scalefactors(args.spatial_dir)
    img_path, sf_key = choose_image(args.spatial_dir, args.use)
    scale = float(sf[sf_key])

    # coordinates on the chosen image resolution
    # NOTE: 10x gives fullres coords; multiply by scale to match hires/lowres image pixels
    x = df["pxl_col_in_fullres"].astype(float).values * scale
    y = df["pxl_row_in_fullres"].astype(float).values * scale
    xy = np.stack([x, y], axis=1)

    bid, rects, (x0,x1,y0,y1) = make_blocks(xy, (gx,gy))

    # select test blocks
    rng = np.random.default_rng(args.seed)
    all_blocks = np.arange(gx*gy)
    n_test_blocks = int(math.ceil(args.test_frac * len(all_blocks)))
    perm = rng.permutation(all_blocks)
    test_blocks = set(perm[:n_test_blocks].tolist())
    is_test = np.array([b in test_blocks for b in bid], dtype=bool)

    # load image
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    ax = plt.gca()

    # draw test block rectangles (semi-transparent)
    for b in test_blocks:
        rx0, ry0, dx, dy = rects[b]
        # clip to image range (optional)
        rect = Rectangle(
            (rx0, ry0), dx, dy,
            linewidth=1.0, edgecolor="yellow",
            facecolor=(1.0, 1.0, 0.0, 0.15)  # RGBA
        )
        ax.add_patch(rect)

    # draw spots
    # train
    plt.scatter(xy[~is_test,0], xy[~is_test,1], s=args.dot_size, alpha=0.7, label="train spots")
    # test
    plt.scatter(xy[is_test,0],  xy[is_test,1],  s=args.dot_size, alpha=0.9, label="test spots")

    plt.title(f"Block split on H&E ({args.use}) | grid={gx}x{gy}, test_frac={args.test_frac}, seed={args.seed}")
    plt.axis("off")
    plt.legend(loc="lower right", frameon=True)

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=220)
    plt.close()
    print("Saved:", args.out_png)
    print(f"spots: {len(df)} | test_blocks: {len(test_blocks)}/{gx*gy} | test_spots: {is_test.sum()}")

if __name__ == "__main__":
    main()

# python /home/xiaoxinyu/代谢/TEXT/within-test-pathway/07_plot_blocksplit_on_he.py \
#   --spatial_dir /home/xiaoxinyu/代谢/scFEA/input/ccRCC/ST_visium/Y27_T/outs/spatial \
#   --out_png /home/xiaoxinyu/代谢/scFEA/output/ccRCC/within_patient_runs/Y27_pathway_blocksplit_v1/fig_split/Y27_blocksplit_hires.png \
#   --grid 6,6 \
#   --test_frac 0.8 \
#   --seed 0 \
#   --use hires \
#   --only_in_tissue
