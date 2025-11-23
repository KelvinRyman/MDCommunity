import os
import csv
import itertools
from typing import Dict, Set, Tuple, Optional


def score_pair(
    n1: int,
    n2: int,
    overlap: int,
    e1: int,
    e2: int,
    max_edges: int,
    size_weight: float = 0.3,
    overlap_weight: float = 0.5,
    repr_weight: float = 0.2,
) -> float:
    """为一对层打分，越大越好。"""
    # 规模相当：相差越小得分越高
    size_score = 0.0 if max(n1, n2) == 0 else 1.0 - abs(n1 - n2) / max(n1, n2)
    # 重叠度：交集节点占较小层的比例
    overlap_score = 0.0 if min(n1, n2) == 0 else overlap / min(n1, n2)
    # 代表性：用边数占比作为简易 proxy
    if max_edges <= 0:
        repr_score = 0.0
    else:
        repr_score = ((e1 / max_edges) + (e2 / max_edges)) / 2.0

    return (
        size_weight * size_score
        + overlap_weight * overlap_score
        + repr_weight * repr_score
    )


def summarize_real_edges(real_dir: str, output_csv: str = None) -> None:
    """
    扫描 data/real 下的 .edges 文件，统计：
      - 文件名（数据集名）
      - 节点总数（最大节点编号，假定文件里节点编号从 1 开始）
      - 层数（不同 layer_id 的数量）
      - 自动选出的两层 (layer_m, layer_n)：依据“规模相近 + 重叠度高 + 代表性（边数多）”的启发式。

    如果提供 output_csv，则把结果写入 CSV，否则打印到标准输出。
    """
    if not os.path.isdir(real_dir):
        raise FileNotFoundError(f"real_dir not found: {real_dir}")

    rows = []

    for fname in sorted(os.listdir(real_dir)):
        if not fname.endswith(".edges"):
            continue
        path = os.path.join(real_dir, fname)
        dataset = fname.replace(".edges", "")

        layer_nodes: Dict[int, Set[int]] = {}
        layer_edges: Dict[int, int] = {}
        max_node = 0

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    layer_id = int(parts[0])
                    u = int(parts[1])
                    v = int(parts[2])
                except ValueError:
                    continue

                s = layer_nodes.setdefault(layer_id, set())
                s.add(u)
                s.add(v)
                layer_edges[layer_id] = layer_edges.get(layer_id, 0) + 1

                if u > max_node:
                    max_node = u
                if v > max_node:
                    max_node = v

        layers = sorted(layer_nodes.keys())
        n_layers = len(layers)
        if n_layers < 2:
            # 单层网络，无法选对
            rows.append(
                {
                    "dataset": dataset,
                    "n_nodes": max_node,
                    "n_layers": n_layers,
                    "layer_m": "",
                    "layer_n": "",
                }
            )
            continue

        max_edges = max(layer_edges.values()) if layer_edges else 0

        best_score = -1.0
        best_pair: Optional[Tuple[int, int]] = None
        for l1, l2 in itertools.combinations(layers, 2):
            nodes1 = layer_nodes[l1]
            nodes2 = layer_nodes[l2]
            overlap = len(nodes1 & nodes2)
            n1 = len(nodes1)
            n2 = len(nodes2)
            e1 = layer_edges.get(l1, 0)
            e2 = layer_edges.get(l2, 0)
            s = score_pair(n1, n2, overlap, e1, e2, max_edges)
            if s > best_score:
                best_score = s
                best_pair = (l1, l2)

        layer_m, layer_n = best_pair if best_pair is not None else ("", "")

        rows.append(
            {
                "dataset": dataset,
                "n_nodes": max_node,
                "n_layers": n_layers,
                "layer_m": layer_m,
                "layer_n": layer_n,
            }
        )

    fieldnames = ["dataset", "n_nodes", "n_layers", "layer_m", "layer_n"]

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        print("\t".join(fieldnames))
        for r in rows:
            print(
                f"{r['dataset']}\t{r['n_nodes']}\t{r['n_layers']}\t"
                f"{r['layer_m']}\t{r['layer_n']}"
            )


if __name__ == "__main__":
    # 从当前脚本推断仓库根目录，然后定位 data/real
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)
    real_dir = os.path.join(repo_root, "data", "real")
    output_csv_path = os.path.join(real_dir, "real_edges_summary.csv")

    summarize_real_edges(real_dir, output_csv=output_csv_path)
    print(f"Summary written to: {output_csv_path}")
