#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, torch
import pandas as pd

def _as_list_str(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().tolist()
    return [str(v) for v in list(x)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", required=True)
    ap.add_argument("--pathway_dir", required=True)
    ap.add_argument("--patients", nargs="+", required=True)
    ap.add_argument("--pathway_suffix", default="_T_SM_Pathway.filtered.spot.intersect.csv")
    args = ap.parse_args()

    for pid in args.patients:
        gfp = os.path.join(args.graph_dir, f"{pid}.pt")
        yfp = os.path.join(args.pathway_dir, f"{pid}{args.pathway_suffix}")

        data = torch.load(gfp, map_location="cpu", weights_only=False)
        g_spots = _as_list_str(data.spot_id)
        y = pd.read_csv(yfp, index_col=0)
        y_spots = list(y.index.astype(str))

        print(f"\n=== {pid} ===")
        print("graph nodes:", len(g_spots), "pathway rows:", len(y_spots))
        assert len(g_spots) == len(y_spots), "length mismatch"
        assert g_spots == y_spots, "order mismatch"
        print(f"[{pid}] ✅ perfectly aligned")

    print("\n🎉 All passed.")

if __name__ == "__main__":
    main()

# python /home/xiaoxinyu/代谢/TEXT/within-test-pathway/01_check_align_anysuffix.py \
#   --graph_dir /home/xiaoxinyu/代谢/scFEA/output/ccRCC/pyg_graphs_v6b_intersect \
#   --pathway_dir /home/xiaoxinyu/代谢/scFEA/input/ccRCC/new_data/pathway_level_v2_filtered_spot_intersect \
#   --patients R114 S15 X49 Y27 Y7
