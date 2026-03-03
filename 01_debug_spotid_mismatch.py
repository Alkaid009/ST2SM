#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import pandas as pd

def _to_list_str(x):
    import torch
    if torch.is_tensor(x):
        x = x.detach().cpu().tolist()
    return [str(v) for v in list(x)]

def normalize_candidates(s: str):
    """
    给出一组常见“可能匹配”的规范化版本，用来猜测映射规则。
    你可以把这里当成 spot_id 规则探测器。
    """
    s = str(s)
    cands = set()
    cands.add(s)

    # 去掉空白
    cands.add(s.strip())

    # 常见分隔符：_ / | / : / space
    for sep in ["_", "|", ":", " "]:
        if sep in s:
            cands.add(s.split(sep)[0])
            cands.add(s.split(sep)[-1])

    # 如果像 "AAAC...-1-1-0-0-..." 这种，取前两段/第一段
    parts = s.split("-")
    if len(parts) >= 2:
        cands.add("-".join(parts[:2]))  # AAAC...-1
    if len(parts) >= 1:
        cands.add(parts[0])             # AAAC...

    # 如果有 ".h5ad" 之类
    if s.endswith(".h5ad"):
        cands.add(os.path.basename(s).replace(".h5ad",""))

    return list(cands)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_pt", required=True)
    ap.add_argument("--pathway_csv", required=True)
    ap.add_argument("--sm_csv", required=False, default=None,
                    help="optional: {pid}_T_SM.csv (the one used to write spot_id)")
    ap.add_argument("--show_n", type=int, default=12)
    args = ap.parse_args()

    data = torch.load(args.graph_pt, map_location="cpu", weights_only=False)
    if not hasattr(data, "spot_id"):
        raise RuntimeError("graph has no spot_id")

    g_spots = _to_list_str(data.spot_id)
    y_df = pd.read_csv(args.pathway_csv, index_col=0)
    y_spots = list(y_df.index.astype(str))

    print("=== Samples ===")
    print("[graph spot_id examples]")
    for s in g_spots[:args.show_n]:
        print(" ", s)
    print("\n[pathway index examples]")
    for s in y_spots[:args.show_n]:
        print(" ", s)

    # 如果给了 SM csv，就也看看 spot_id 长啥样
    if args.sm_csv is not None:
        sm = pd.read_csv(args.sm_csv, usecols=["spot_id"])
        sm_spots = sm["spot_id"].astype(str).tolist()
        print("\n[SM csv spot_id examples]")
        for s in sm_spots[:args.show_n]:
            print(" ", s)

    # ====== mapping probe ======
    g_set = set(g_spots)
    y_set = set(y_spots)

    direct_hit = len(y_set & g_set)
    print("\n=== Direct overlap ===")
    print(f"direct overlap count = {direct_hit} / pathway_spots={len(y_spots)}")

    # 尝试用 normalize_candidates 进行“软匹配”估计：pathway -> graph
    g_norm_set = set()
    for gs in g_spots:
        for c in normalize_candidates(gs):
            g_norm_set.add(c)

    soft_hit = 0
    miss_examples = []
    for ys in y_spots:
        ys_cands = normalize_candidates(ys)
        ok = any(c in g_norm_set for c in ys_cands)
        if ok:
            soft_hit += 1
        else:
            if len(miss_examples) < 20:
                miss_examples.append(ys)

    print("\n=== Soft overlap (by normalization candidates) ===")
    print(f"soft-hit count = {soft_hit} / pathway_spots={len(y_spots)}")
    if miss_examples:
        print("miss examples (first 20):")
        for s in miss_examples:
            print(" ", s)

    # 反向：graph -> pathway 也试一下
    y_norm_set = set()
    for ys in y_spots:
        for c in normalize_candidates(ys):
            y_norm_set.add(c)

    soft_hit2 = 0
    for gs in g_spots:
        gs_cands = normalize_candidates(gs)
        if any(c in y_norm_set for c in gs_cands):
            soft_hit2 += 1
    print("\n=== Soft overlap (graph -> pathway) ===")
    print(f"soft-hit count = {soft_hit2} / graph_spots={len(g_spots)}")

if __name__ == "__main__":
    main()

# python /home/xiaoxinyu/代谢/TEXT/within-test-pathway/01_debug_spotid_mismatch.py \
#   --graph_pt /home/xiaoxinyu/代谢/scFEA/output/ccRCC/pyg_graphs_v5_spotid_from_sm/R114.pt \
#   --pathway_csv /home/xiaoxinyu/代谢/scFEA/input/ccRCC/new_data/pathway_level_v2_filtered_spot/R114_T_SM_Pathway.filtered.spot.csv \
#   --sm_csv /home/xiaoxinyu/代谢/scFEA/input/ccRCC/new_data/R114_T_SM.csv
