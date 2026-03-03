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
    ap.add_argument("--out_dir", required=True, help="output dir for subgraphs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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

        # 目标节点：按 pathway 的顺序排列（保持 y 的 row order）
        missing = [s for s in y_spots if s not in spot2idx]
        if len(missing) > 0:
            raise RuntimeError(f"[{pid}] {len(missing)} pathway spots not found in graph. e.g. {missing[:5]}")

        node_idx = torch.tensor([spot2idx[s] for s in y_spots], dtype=torch.long)

        # 生成子图：relabel_nodes=True -> 新图节点编号从 0..N-1
        edge_index = data.edge_index
        new_edge_index, _ = subgraph(node_idx, edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

        # 构造新 Data（保留常用属性：x/pos/spot_id）
        new_data = data.__class__()  # torch_geometric.data.Data()
        # 必需字段
        new_data.edge_index = new_edge_index
        if hasattr(data, "x") and data.x is not None:
            new_data.x = data.x[node_idx]
        if hasattr(data, "pos") and data.pos is not None:
            new_data.pos = data.pos[node_idx]
        # 关键：spot_id 与 y_df 完全一致
        new_data.spot_id = y_spots

        # 可选：复制其它你可能会用到的字段（edge_attr 等）
        for k in ["edge_attr", "batch"]:
            if hasattr(data, k):
                setattr(new_data, k, getattr(data, k))

        # 保存 idx 映射，便于排查
        map_fp = os.path.join(args.out_dir, f"{pid}.nodeidx_map.tsv")
        with open(map_fp, "w", encoding="utf-8") as f:
            f.write("new_idx\told_idx\tspot_id\n")
            for new_i, old_i in enumerate(node_idx.tolist()):
                f.write(f"{new_i}\t{old_i}\t{y_spots[new_i]}\n")

        out_fp = os.path.join(args.out_dir, f"{pid}.pt")
        torch.save(new_data, out_fp)

        print(f"\n[{pid}] ✅ subgraph saved -> {out_fp}")
        print(f"  old num_nodes={data.num_nodes}, new num_nodes={len(y_spots)}")
        print(f"  x: {tuple(new_data.x.shape) if hasattr(new_data,'x') else None}, pos: {tuple(new_data.pos.shape) if hasattr(new_data,'pos') else None}")
        print(f"  edges: {new_data.edge_index.shape[1]}")
        print(f"  node map -> {map_fp}")

if __name__ == "__main__":
    main()

# python /home/xiaoxinyu/代谢/TEXT/within-test-pathway/01_make_subgraph_match_pathway.py \
#   --graph_dir /home/xiaoxinyu/代谢/scFEA/output/ccRCC/pyg_graphs_v5_spotid_from_sm \
#   --pathway_dir /home/xiaoxinyu/代谢/scFEA/input/ccRCC/new_data/pathway_level_v2_filtered_spot \
#   --patients R114 S15 X49 Y27 Y7 \
#   --out_dir /home/xiaoxinyu/代谢/scFEA/output/ccRCC/pyg_graphs_v6_match_pathway_spots
