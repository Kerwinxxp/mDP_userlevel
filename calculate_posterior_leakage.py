# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, List, Optional
import argparse
from pathlib import Path

def resolve_path(p: str) -> str:
    """Resolve relative paths relative to this script's directory."""
    pth = Path(p)
    if pth.is_absolute():
        return str(pth.resolve())
    return str((Path(__file__).resolve().parent / pth).resolve())
# ===================== Experiment knobs (keep minimal) =====================
DEFAULT_VIOLATION_THRESHOLD = 5.0 # 你只需要经常改这个（或用 CLI 覆盖）
SAVE_DIR_DEFAULT = "posterior_leakage_results"
DEFAULT_CLASS_METRIC_FILE = "distance_matrix_avg.json"

# NEW: force prior to uniform (can be overridden by CLI)
DEFAULT_UNIFORM_PRIOR = False

# NEW: merged summary output (like all_new.txt)
DEFAULT_MERGED_SUMMARY = True
DEFAULT_MERGED_SUMMARY_NAME = "all_summary_7_noiseonly.txt"

# numeric / plotting constants (usually don't need to change)
EPS = 1e-12
DIAG_FILL = 1e-12
HIST_BINS = 50

# NEW: make distribution milder when converting logits -> probs
SOFTMAX_T = 3.0  # try 2/3/5/10; larger => flatter (more "gentle")


def softmax_temp(z: np.ndarray, T: float = 1.0, axis: int = -1) -> np.ndarray:
    """Temperature softmax. T>1 flattens, T<1 sharpens. Supports vectors/matrices."""
    if T is None:
        T = 1.0
    T = float(T)
    if T <= 0:
        raise ValueError("Temperature T must be > 0")

    z = np.asarray(z, dtype=float) / T
    z = z - np.max(z, axis=axis, keepdims=True)  # stable
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# ===================== 聚合与对齐 =====================
def aggregate_by_label_logits(logits: np.ndarray, labels: np.ndarray, T: float = SOFTMAX_T):
    """
    将同一 label 的多个窗口级 logits 相加（≈独立证据相乘），
    再 temperature-softmax 得到“对象级”概率分布。

    重要说明：
    - 这里返回的 probs 是 softmax_temp(sum(logits), T) 的结果（对象级）。
    - T>1 会让分布更“温和/平坦”，减弱最大类的放大效应。
    返回：agg_probs (num_labels, C), agg_labels (num_labels,)
    """
    labels = labels.astype(int)
    uniq = np.unique(labels)
    agg_logits = []
    agg_labels = []
    for lb in uniq:
        L = logits[labels == lb]          # (k, C)
        summed = L.sum(axis=0)            # logits 相加
        agg_logits.append(summed)
        agg_labels.append(lb)
    agg_logits = np.vstack(agg_logits)    # (num_labels, C)

    probs = softmax_temp(agg_logits, T=T, axis=1)
    return probs, np.array(agg_labels)


def aggregate_by_label_probs_mult(probs: np.ndarray, labels: np.ndarray, eps: Optional[float] = None):
    """
    备选：当没有 logits 时，把同一 label 的窗口级概率相乘（log 概率相加）再归一化。
    返回：agg_probs (num_labels, C), agg_labels (num_labels,)
    """
    if eps is None:
        eps = EPS

    labels = labels.astype(int)
    uniq = np.unique(labels)
    P = np.clip(probs, eps, 1.0)
    agg_logp = []
    agg_labels = []
    for lb in uniq:
        block = P[labels == lb]           # (k, C)
        logp = np.log(block).sum(axis=0)  # 概率乘积 → log 概率相加
        agg_logp.append(logp)
        agg_labels.append(lb)
    agg_logp = np.vstack(agg_logp)

    # log-sum-exp 归一化
    m = np.max(agg_logp, axis=1, keepdims=True)
    ex = np.exp(agg_logp - m)
    agg_probs = ex / ex.sum(axis=1, keepdims=True)
    return agg_probs, np.array(agg_labels)


def align_by_label_after_aggregation(
    prior_probs: np.ndarray, prior_labels: np.ndarray,
    posterior_probs: np.ndarray, posterior_labels: np.ndarray
):
    """
    假设两侧都已按 label 聚合为对象级分布。
    对共同的 label 排序后对齐，保证同一 label 在 A/B 同一行。
    """
    common = np.intersect1d(np.unique(prior_labels), np.unique(posterior_labels))
    if common.size == 0:
        raise ValueError("No common labels after aggregation; cannot align prior/posterior.")

    prior_idx = {lb: i for i, lb in enumerate(prior_labels)}
    post_idx  = {lb: i for i, lb in enumerate(posterior_labels)}
    A, B, L = [], [], []
    for lb in sorted(common):
        A.append(prior_probs[prior_idx[lb]])
        B.append(posterior_probs[post_idx[lb]])
        L.append(lb)
    return np.vstack(A), np.vstack(B), np.array(L)


# ===================== Pairwise mPL 计算 =====================

def calculate_pairwise_posterior_leakage(
    prior_probs: np.ndarray,
    posterior_probs: np.ndarray,
    class_metric: np.ndarray = None,
    epsilon: Optional[float] = None,
    batch_size: int = 2048,
) -> Dict[str, Any]:
    """
    逐对 (i,j) 类别计算 | log(B_i/B_j) - log(A_i/A_j) | / d_{i,j}

    固定只统计无序对 i<j（只算 (i,j)，不重复算 (j,i)）。

    性能优化：
    - 仅在上三角 pairs 上计算（使用 np.triu_indices），避免构造 C×C 的 diff 矩阵
    - 对对象维度按 batch 处理，减少 Python 循环与临时大数组
    """
    if epsilon is None:
        epsilon = EPS

    assert prior_probs.shape == posterior_probs.shape, "prior/posterior 维度不一致"
    N, C = prior_probs.shape

    A = np.clip(prior_probs, epsilon, 1.0)
    B = np.clip(posterior_probs, epsilon, 1.0)
    A = A / A.sum(axis=1, keepdims=True)
    B = B / B.sum(axis=1, keepdims=True)

    # 只取 i<j 的 pair 索引（P = C*(C-1)/2）
    i_idx, j_idx = np.triu_indices(C, k=1)
    P = int(i_idx.size)

    # 准备距离分母（只取 pairs 部分）
    if class_metric is None:
        denom = np.ones((P,), dtype=float)
    else:
        D = np.array(class_metric, dtype=float)
        assert D.shape == (C, C)
        denom = D[i_idx, j_idx].astype(float, copy=False)
        denom = np.where(denom <= 0, epsilon, denom)

    # 预计算 log，避免每个对象重复 np.log
    logA = np.log(A)
    logB = np.log(B)

    # batch 化计算，避免 N 次 Python 循环 + 避免构造 (C,C) 矩阵
    if batch_size is None or batch_size <= 0:
        batch_size = N  # 一次性（可能占内存）

    chunks: List[np.ndarray] = []
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        la = logA[start:end]  # (b, C)
        lb = logB[start:end]  # (b, C)

        # (b, P): 只在 pairs 上做差
        delta = (lb[:, i_idx] - lb[:, j_idx]) - (la[:, i_idx] - la[:, j_idx])
        pl = np.abs(delta) / denom  # broadcast denom: (P,)
        chunks.append(pl.reshape(-1))

    arr = np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=float)
    pairwise_pl_values = arr.tolist()

    stats = {
        'mean_pl': float(np.mean(arr)) if arr.size else 0.0,
        'std_pl': float(np.std(arr)) if arr.size else 0.0,
        'min_pl': float(np.min(arr)) if arr.size else 0.0,
        'max_pl': float(np.max(arr)) if arr.size else 0.0,
        'median_pl': float(np.median(arr)) if arr.size else 0.0,
        'total_counts': int(arr.size)
    }
    return {'pairwise_pl': pairwise_pl_values, 'statistics': stats}


# ===================== 可视化（pairwise 版本） =====================

def create_pl_distribution_plot(
    pl_values: List[float],
    save_path: str,
    dataset_comparison: str,
    violation_threshold: Optional[float] = None
):
    """把每个 (i,j) 的 PL 当作一个样本/计数来画直方图"""
    if violation_threshold is None:
        violation_threshold = DEFAULT_VIOLATION_THRESHOLD

    arr = np.array(pl_values)
    if len(arr) == 0:
        print("No PL values to plot")
        return

    plt.figure(figsize=(12, 8))

    counts, bin_edges = np.histogram(arr, bins=HIST_BINS)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    not_violated_mask = bin_centers <= violation_threshold
    violated_mask = bin_centers > violation_threshold

    plt.bar(bin_centers[not_violated_mask], counts[not_violated_mask],
            width=bin_width, alpha=0.7, edgecolor='black',
            color='steelblue', label='Not violated')

    if np.any(violated_mask):
        plt.bar(bin_centers[violated_mask], counts[violated_mask],
                width=bin_width, alpha=0.7, edgecolor='black',
                color='red', label='Violated')

    plt.axvline(violation_threshold, color='darkred', linestyle='-', linewidth=3,
                label=f'Violation threshold (ε = {violation_threshold})')
    plt.axvline(np.mean(arr), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(arr):.4f}')

    violated_values = arr[arr > violation_threshold]
    violation_ratio = len(violated_values) / len(arr) * 100 if len(arr) > 0 else 0.0

    plt.xlabel('Pairwise Posterior Leakage (PL)', fontsize=12)
    plt.ylabel('Count (number of class pairs)', fontsize=12)
    plt.title(f'Posterior Leakage Distribution (pairwise)\n{dataset_comparison}\nViolation ratio: {violation_ratio:.1f}%',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Pairwise PL distribution plot saved: {save_path}")


# ===================== 主流程 =====================

def analyze_posterior_leakage_between_datasets(
    prior_file: str,
    posterior_file: str,
    save_dir: str = SAVE_DIR_DEFAULT,
    class_metric: np.ndarray = None,
    violation_threshold: Optional[float] = None,
    uniform_prior: bool = DEFAULT_UNIFORM_PRIOR,
    batch_size: int = 2048,
):
    """
    分析两个数据集之间的后验概率泄露（pairwise 统计）：
    - 窗口级 -> （按 label 聚合）对象级概率
    - 对同一对象的所有类对 (i,j) 产生一个 PL 值（count=1）
    - 直方图按所有 pairwise PL 值作图
    """
    if violation_threshold is None:
        violation_threshold = DEFAULT_VIOLATION_THRESHOLD

    print(f"\n{'='*60}")
    print(f"分析后验泄露（pairwise 成对赔率定义）")
    print(f"Prior (未加噪): {prior_file}")
    print(f"Posterior (加噪): {posterior_file}")
    print(f"{'='*60}\n")

    # 加载数据
    with open(prior_file, 'r', encoding='utf-8') as f:
        prior_data = json.load(f)
    with open(posterior_file, 'r', encoding='utf-8') as f:
        posterior_data = json.load(f)

    prior_labels_raw = np.array(prior_data['labels'])
    posterior_labels_raw = np.array(posterior_data['labels'])

    # --- 优先使用 logits 做“联合观测”聚合 ---
    # 若提供 logits：使用 softmax(sum logits) 得到对象级概率；
    # 否则：使用输入的窗口级 probs 做概率乘积近似聚合。
    if 'logits' in prior_data and 'logits' in posterior_data:
        prior_logits_raw = np.array(prior_data['logits'])
        posterior_logits_raw = np.array(posterior_data['logits'])

        prior_probs_agg, prior_labels_agg = aggregate_by_label_logits(prior_logits_raw, prior_labels_raw, T=SOFTMAX_T)
        posterior_probs_agg, posterior_labels_agg = aggregate_by_label_logits(posterior_logits_raw, posterior_labels_raw, T=SOFTMAX_T)
        aggregation_method = "logits_sum_softmax_temp"
        print(f"已使用 logits 聚合为对象级概率（temperature-softmax, T={SOFTMAX_T}）。")
    else:
        prior_probs_raw = np.array(prior_data['probs'])
        posterior_probs_raw = np.array(posterior_data['probs'])

        prior_probs_agg, prior_labels_agg = aggregate_by_label_probs_mult(prior_probs_raw, prior_labels_raw)
        posterior_probs_agg, posterior_labels_agg = aggregate_by_label_probs_mult(posterior_probs_raw, posterior_labels_raw)
        aggregation_method = "prob_product_norm"
        print("未发现 logits，已使用概率乘积近似聚合为对象级概率。")

    # --- 对齐（对象级）---
    prior_probs, posterior_probs, labels = align_by_label_after_aggregation(
        prior_probs_agg, prior_labels_agg,
        posterior_probs_agg, posterior_labels_agg
    )

    # NEW: override prior with uniform distribution (keep posterior unchanged)
    if uniform_prior:
        C_ = int(posterior_probs.shape[1])
        prior_probs = np.full((prior_probs.shape[0], C_), 1.0 / C_, dtype=float)
        print(f"✅ 已将 prior 强制设为 uniform 分布（每类=1/{C_}），其余保持不变。")

    print(f"对齐后对象数：{len(labels)}")
    print(f"类别数：{prior_probs.shape[1]}")

    # 固定 pair 模式（只算 i<j）
    C = int(prior_probs.shape[1])
    pairs_per_object = (C * (C - 1)) // 2
    pair_mode = "unordered(i<j)"

    # 计算 pairwise 后验泄露
    print("计算 pairwise 后验泄露（同一对象内成对类赔率变化）...")
    pl_result = calculate_pairwise_posterior_leakage(
        prior_probs,
        posterior_probs,
        class_metric=class_metric,
        epsilon=EPS,
        batch_size=batch_size,
    )

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 提取数据集名称
    prior_name = os.path.basename(prior_file).replace('per_sample_probs_', '').replace('.json', '')
    posterior_name = os.path.basename(posterior_file).replace('per_sample_probs_', '').replace('.json', '')
    comparison_name = f"{prior_name}_vs_{posterior_name}"

    # 保存详细结果
    detailed_result = {
        'comparison': {
            'prior_dataset': prior_name,
            'posterior_dataset': posterior_name,
            'num_aligned_objects': int(len(labels)),
            'num_classes': C,
            'aggregation': aggregation_method,
            'uniform_prior': bool(uniform_prior),
            'softmax_temperature_T': float(SOFTMAX_T) if aggregation_method == "logits_sum_softmax_temp" else None,
            'pair_mode': pair_mode,
            'pairs_per_object': int(pairs_per_object),
            'pairwise_total_counts': pl_result['statistics']['total_counts']
        },
        'pairwise_posterior_leakage': {
            'pairwise_pl': pl_result['pairwise_pl'],
            'statistics': pl_result['statistics']
        }
    }

    detailed_path = os.path.join(save_dir, f"{comparison_name}_pairwise_leakage_detailed.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_result, f, indent=2, ensure_ascii=False)
    print(f"✅ 详细结果已保存: {detailed_path}")

    # 创建分布图（pairwise）
    distribution_path = os.path.join(save_dir, f"{comparison_name}_pairwise_distribution.png")
    create_pl_distribution_plot(
        pl_result['pairwise_pl'],
        distribution_path,
        f"{prior_name} (prior) vs {posterior_name} (posterior)",
        violation_threshold=violation_threshold
    )

    # 生成摘要
    stats = pl_result['statistics']
    summary_path = os.path.join(save_dir, f"{comparison_name}_pairwise_summary.txt")

    # NEW: build summary text once (for both per-run file and merged file)
    summary_lines = []
    summary_lines.append("Posterior Leakage Analysis Summary (Pairwise mPL)\n")
    summary_lines.append("=" * 50 + "\n")
    summary_lines.append(f"Prior Dataset: {prior_name}\n")
    summary_lines.append(f"Posterior Dataset: {posterior_name}\n")
    summary_lines.append(f"Aligned Objects: {len(labels)}\n")
    summary_lines.append(f"Number of Classes: {prior_probs.shape[1]}\n")
    summary_lines.append(f"Aggregation: {aggregation_method}\n")
    summary_lines.append(f"Uniform prior: {bool(uniform_prior)}\n")
    if aggregation_method == "logits_sum_softmax_temp":
        summary_lines.append(f"Softmax temperature T: {SOFTMAX_T}\n")
    summary_lines.append(f"Pair mode: {pair_mode}\n")
    summary_lines.append(f"Pairs per object: {pairs_per_object}\n\n")
    summary_lines.append("Pairwise Posterior Leakage Statistics:\n")
    summary_lines.append("-" * 30 + "\n")
    summary_lines.append(f"Total Counts (pairs across all objects): {stats['total_counts']}\n")
    summary_lines.append(f"Mean PL: {stats['mean_pl']:.6f}\n")
    summary_lines.append(f"Std PL: {stats['std_pl']:.6f}\n")
    summary_lines.append(f"Median PL: {stats['median_pl']:.6f}\n")
    summary_lines.append(f"Min PL: {stats['min_pl']:.6f}\n")
    summary_lines.append(f"Max PL: {stats['max_pl']:.6f}\n")
    summary_text = "".join(summary_lines)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"✅ 摘要已保存: {summary_path}")

    # 打印统计信息
    print(f"\n{'='*60}")
    print("后验泄露统计 (Pairwise mPL):")
    print(f"{'='*60}")
    print(f"Counts: {stats['total_counts']}")
    print(f"平均 PL: {stats['mean_pl']:.6f}")
    print(f"标准差 PL: {stats['std_pl']:.6f}")
    print(f"中位数 PL: {stats['median_pl']:.6f}")
    print(f"最小 PL: {stats['min_pl']:.6f}")
    print(f"最大 PL: {stats['max_pl']:.6f}")
    print(f"{'='*60}\n")

    # NEW: attach summary text for merged output
    detailed_result["summary_text"] = summary_text
    return detailed_result


# ===================== CLI =====================

if __name__ == "__main__":
    # ===================== 如何运行（直接复制即可） =====================
    # 1) 默认：一次跑完 noise_1..noise_10（prior 固定为 budget_7_mask_abstract.json）
    #    python calculate_posterior_leakage.py
    #
    # 2) 只跑某一个（例如 noise_3）
    #    python calculate_posterior_leakage.py --noise-start 3 --noise-end 3
    #
    # 3) 跑自定义范围（例如 noise_4..noise_8）
    #    python calculate_posterior_leakage.py --noise-start 4 --noise-end 8
    #
    # 4) posterior 文件名不符合模板时：直接给列表
    #    python calculate_posterior_leakage.py --posterior-files budget_7_noise_3_abstract.json budget_7_noise_7_abstract.json
    #
    # 5) 指定 prior / 输出目录 / 阈值：
    #    python calculate_posterior_leakage.py --prior-file budget_7_mask_abstract.json --save-dir posterior_leakage_results --threshold 2.0
    # ===============================================================

    files = [
        'outputs/WikiActors/multi_7_noiseonly/eval_results_noise_0p001_abstract.json',
        'outputs/WikiActors/multi_7_noiseonly/eval_results_noise_3p0_abstract.json',
    ]

    parser = argparse.ArgumentParser(description="Pairwise posterior leakage analysis (i<j only)")
    parser.add_argument("--prior-file", default=files[0])
    parser.add_argument("--posterior-file", default=files[1])

    # NEW: 允许指定输出目录（可选；默认 SAVE_DIR_DEFAULT）
    parser.add_argument("--save-dir", default=SAVE_DIR_DEFAULT, help="Directory to save results (relative to this script if not absolute).")

    # 批量 posterior（两种方式二选一）
    parser.add_argument(
        "--posterior-files",
        nargs="*",
        default=None,
        help="Optional: provide multiple posterior json files. If set, overrides --posterior-file/template/range.",
    )
    parser.add_argument(
        "--posterior-template",
        type=str,
        default="outputs/WikiActors/multi_7_noiseonly/eval_results_noise_{i}p0_abstract.json",

        # help='Optional: template to generate posterior files, e.g. "budget_2_clean_noise_{i}p0_abstract.json".',
    )
    parser.add_argument("--noise-start", type=int, default=1, help="Start index i for --posterior-template.")
    parser.add_argument("--noise-end", type=int, default=10, help="End index i for --posterior-template (inclusive).")

    parser.add_argument(
        "--threshold",
        dest="violation_threshold",
        type=float,
        default=DEFAULT_VIOLATION_THRESHOLD,
        help="Violation threshold for PL histogram coloring/ratio.",
    )

    # NEW: toggle uniform prior
    parser.add_argument("--uniform-prior", dest="uniform_prior", action="store_true",
                        help="Force prior distribution A to be uniform after alignment.")
    parser.add_argument("--original-prior", dest="uniform_prior", action="store_false",
                        help="Use prior distribution from file (after aggregation/alignment).")
    parser.set_defaults(uniform_prior=DEFAULT_UNIFORM_PRIOR)

    # NEW: merged summary options
    parser.add_argument("--merged-summary", dest="merged_summary", action="store_true",
                        help="Write all per-run summaries into one merged txt under --save-dir.")
    parser.add_argument("--no-merged-summary", dest="merged_summary", action="store_false",
                        help="Disable merged summary txt.")
    parser.set_defaults(merged_summary=DEFAULT_MERGED_SUMMARY)

    parser.add_argument("--merged-summary-name", type=str, default=DEFAULT_MERGED_SUMMARY_NAME,
                        help="Filename for merged summary txt (saved inside --save-dir).")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size over objects for pairwise PL computation. Larger is faster but uses more RAM.",
    )

    args = parser.parse_args()

    # NEW: 统一把相对路径解析到“脚本所在目录”，保证直接运行稳定
    args.prior_file = resolve_path(args.prior_file)
    args.posterior_file = resolve_path(args.posterior_file)
    args.save_dir = resolve_path(args.save_dir)

    # 固定：尝试加载默认类别距离矩阵文件；不存在则退化为 unit metric
    try:
        metric_path = resolve_path(DEFAULT_CLASS_METRIC_FILE)
        with open(metric_path, "r", encoding="utf-8") as f:
            distance_data = json.load(f)
        class_metric = np.array(distance_data["distance_matrix"])
        print(f"✅ 加载类别距离矩阵: {class_metric.shape} ({metric_path})")
    except FileNotFoundError:
        print("⚠️ 未找到类别距离矩阵，使用默认值 (全1)")
        class_metric = None

    # 组装 posterior 列表（优先 --posterior-files；否则用 template + range；否则退回单文件）
    if args.posterior_files is not None and len(args.posterior_files) > 0:
        posterior_list = [resolve_path(p) for p in args.posterior_files]
        src = "--posterior-files"
    else:
        if args.noise_end < args.noise_start:
            raise ValueError("--noise-end must be >= --noise-start")
        posterior_list = [resolve_path(args.posterior_template.format(i=i)) for i in range(args.noise_start, args.noise_end + 1)]
        src = "--posterior-template/--noise-start/--noise-end"

        # 保持兼容：如果你显式只想跑单个文件，也可以直接传 --posterior-file
        default_template = "budget_7_noise_{i}_abstract.json"
        if (
            args.posterior_template == default_template
            and args.noise_start == 1
            and args.noise_end == 10
            and args.posterior_file != resolve_path(files[1])
        ):
            posterior_list = [args.posterior_file]
            src = "--posterior-file (single)"

    print(f"\n将批量计算 posterior 文件（来源: {src}），数量={len(posterior_list)}")
    print("Posterior list preview:", posterior_list[:5], ("..." if len(posterior_list) > 5 else ""))

    ok, skipped = 0, 0

    # NEW: open merged summary file once (overwrite), append each run in order
    merged_f = None
    merged_path = None
    if args.merged_summary:
        os.makedirs(args.save_dir, exist_ok=True)
        merged_path = os.path.join(args.save_dir, args.merged_summary_name)
        merged_f = open(merged_path, "w", encoding="utf-8")

    try:
        for posterior_file in posterior_list:
            if not os.path.exists(posterior_file):
                print(f"⚠️ 跳过不存在的 posterior 文件: {posterior_file}")
                skipped += 1
                continue

            result = analyze_posterior_leakage_between_datasets(
                prior_file=args.prior_file,
                posterior_file=posterior_file,
                save_dir=args.save_dir,
                class_metric=class_metric,
                violation_threshold=args.violation_threshold,
                uniform_prior=args.uniform_prior,
                batch_size=args.batch_size,
            )

            if merged_f is not None:
                merged_f.write(result.get("summary_text", ""))
                merged_f.write("\n\n")  # separate blocks like all_new.txt

            ok += 1
    finally:
        if merged_f is not None:
            merged_f.close()
            print(f"✅ 汇总摘要已保存: {merged_path}")

    print(f"\n🎉 批量分析完成: ok={ok}, skipped={skipped}")

# python calculate_posterior_leakage.py --original-prior