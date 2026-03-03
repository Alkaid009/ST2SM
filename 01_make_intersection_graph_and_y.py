#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import pandas as pd
from torch_geometric.utils import subgraph

def _as_list_str(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().tolist()
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            x = x.tolist()
    except Exception:
        pass
    if isinstance(x, str):
        return [x]
    return [str(v) for v in list(x)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", required=True, help="pyg_graphs_v5_spotid_from_sm/")
    ap.add_argument("--pathway_dir", required=True, help="pathway_level_v2_filtered_spot/")
    ap.add_argument("--patients", nargs="+", required=True)

    ap.add_argument("--out_graph_dir", required=True)
    ap.add_argument("--out_y_dir", required=True)
    ap.add_argument("--out_report_dir", required=True)

    ap.add_argument("--min_intersection", type=int, default=1000,
                    help="sanity: intersection size must be >= this")
    args = ap.parse_args()

    os.makedirs(args.out_graph_dir, exist_ok=True)
    os.makedirs(args.out_y_dir, exist_ok=True)
    os.makedirs(args.out_report_dir, exist_ok=True)

    for pid in args.patients:
        graph_fp = os.path.join(args.graph_dir, f"{pid}.pt")
        y_fp = os.path.join(args.pathway_dir, f"{pid}_T_SM_Pathway.filtered.spot.csv")

        if not os.path.exists(graph_fp):
            raise FileNotFoundError(graph_fp)
        if not os.path.exists(y_fp):
            raise FileNotFoundError(y_fp)

        data = torch.load(graph_fp, map_location="cpu", weights_only=False)
        if not hasattr(data, "spot_id"):
            raise RuntimeError(f"[{pid}] graph has no spot_id")

        g_spots = _as_list_str(data.spot_id)
        spot2idx = {s: i for i, s in enumerate(g_spots)}

        y_df = pd.read_csv(y_fp, index_col=0)
        y_spots = list(y_df.index.astype(str))

        inter_spots = [s for s in y_spots if s in spot2idx]  # keep y order
        inter_n = len(inter_spots)

        # reports
        missing_in_graph = [s for s in y_spots if s not in spot2idx]
        extra_in_graph = [s for s in g_spots if s not in set(y_spots)]

        rep_fp = os.path.join(args.out_report_dir, f"{pid}.missing_report.txt")
        with open(rep_fp, "w", encoding="utf-8") as f:
            f.write(f"pid={pid}\n")
            f.write(f"graph_spots={len(g_spots)}\n")
            f.write(f"pathway_spots={len(y_spots)}\n")
            f.write(f"intersection={inter_n}\n")
            f.write(f"missing_in_graph={len(missing_in_graph)}\n")
            f.write(f"extra_in_graph_vs_pathway={len(extra_in_graph)}\n")
            f.write("\n# missing_in_graph examples (first 50)\n")
            for s in missing_in_graph[:50]:
                f.write(s + "\n")

        if inter_n < args.min_intersection:
            raise RuntimeError(f"[{pid}] intersection too small: {inter_n} < {args.min_intersection}. See {rep_fp}")

        # filter y
        y_int = y_df.loc[inter_spots].copy()

        # build node_idx for subgraph in y order
        node_idx = torch.tensor([spot2idx[s] for s in inter_spots], dtype=torch.long)

        new_edge_index, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

        # create new Data
        new_data = data.__class__()
        new_data.edge_index = new_edge_index
        if hasattr(data, "x") and data.x is not None:
            new_data.x = data.x[node_idx]
        if hasattr(data, "pos") and data.pos is not None:
            new_data.pos = data.pos[node_idx]
        new_data.spot_id = inter_spots

        for k in ["edge_attr", "batch"]:
            if hasattr(data, k):
                setattr(new_data, k, getattr(data, k))

        out_g = os.path.join(args.out_graph_dir, f"{pid}.pt")
        out_y = os.path.join(args.out_y_dir, f"{pid}_T_SM_Pathway.filtered.spot.intersect.csv")

        torch.save(new_data, out_g)
        y_int.to_csv(out_y)

        print(f"\n[{pid}] ✅ intersection graph+y saved")
        print(f"  intersection={inter_n} (pathway {len(y_spots)} -> {inter_n})")
        print(f"  graph nodes {len(g_spots)} -> {inter_n}")
        print(f"  out_graph: {out_g}")
        print(f"  out_y    : {out_y}")
        print(f"  report   : {rep_fp}")

if __name__ == "__main__":
    main()

# python /home/xiaoxinyu/代谢/TEXT/within-test-pathway/01_make_intersection_graph_and_y.py \
#   --graph_dir /home/xiaoxinyu/代谢/scFEA/output/ccRCC/pyg_graphs_v5_spotid_from_sm \
#   --pathway_dir /home/xiaoxinyu/代谢/scFEA/input/ccRCC/new_data/pathway_level_v2_filtered_spot \
#   --patients R114 S15 X49 Y27 Y7 \
#   --out_graph_dir /home/xiaoxinyu/代谢/scFEA/output/ccRCC/pyg_graphs_v6b_intersect \
#   --out_y_dir /home/xiaoxinyu/代谢/scFEA/input/ccRCC/new_data/pathway_level_v2_filtered_spot_intersect \
#   --out_report_dir /home/xiaoxinyu/代谢/scFEA/output/ccRCC/pyg_graphs_v6b_intersect_reports
