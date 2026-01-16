# -*- coding: utf-8 -*-
"""
PIInum vs Posterior Leakage 综合分析
包含两种可视化：
1. 多指标散点图 (Mean/Max/Median/95th Percentile PL)
2. 分组箱线图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import os

# ===================== numeric knobs =====================
EPS = 1e-12
SOFTMAX_T = 3.0


def softmax_temp(z: np.ndarray, T: float = 1.0, axis: int = -1) -> np.ndarray:
    """Temperature softmax. T>1 flattens, T<1 sharpens."""
    if T is None:
        T = 1.0
    T = float(T)
    if T <= 0:
        raise ValueError("Temperature T must be > 0")
    z = np.asarray(z, dtype=float) / T
    z = z - np.max(z, axis=axis, keepdims=True)  # stable
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# ===================== 数据加载 =====================

def load_pii_counts(data_file: str) -> Dict[str, int]:
    """从数据文件加载每个 individual 的 PIInum"""
    df = pd.read_json(data_file)
    name_col = 'Name' if 'Name' in df.columns else ('name' if 'name' in df.columns else None)
    if name_col is None or 'PIInum' not in df.columns:
        raise ValueError("数据文件必须包含 'Name' 和 'PIInum' 列")
    pii_dict = dict(zip(df[name_col], df['PIInum']))
    print(f"✅ 从 {data_file} 加载了 {len(pii_dict)} 个样本的 PIInum")
    return pii_dict


def load_label_mapping(label_mapping_file: str) -> Dict[int, str]:
    """从 label_mapping.json 加载 label_to_name"""
    with open(label_mapping_file, "r", encoding="utf-8") as f:
        m = json.load(f)
    if "label_to_name" not in m:
        raise ValueError("label_mapping.json missing 'label_to_name'")
    return {int(k): str(v) for k, v in m["label_to_name"].items()}


# ===================== 聚合函数 =====================

def aggregate_by_label_logits(logits: np.ndarray, labels: np.ndarray, T: float = SOFTMAX_T):
    """聚合窗口级 logits 为对象级概率"""
    labels = labels.astype(int)
    uniq = np.unique(labels)
    agg_logits = []
    agg_labels = []
    for lb in uniq:
        L = logits[labels == lb]
        summed = L.sum(axis=0)
        agg_logits.append(summed)
        agg_labels.append(lb)
    agg_logits = np.vstack(agg_logits)
    probs = softmax_temp(agg_logits, T=T, axis=1)
    return probs, np.array(agg_labels)


def aggregate_by_label_probs_mult(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12):
    """聚合窗口级概率为对象级概率"""
    labels = labels.astype(int)
    uniq = np.unique(labels)
    P = np.clip(probs, eps, 1.0)
    agg_logp = []
    agg_labels = []
    for lb in uniq:
        block = P[labels == lb]
        logp = np.log(block).sum(axis=0)
        agg_logp.append(logp)
        agg_labels.append(lb)
    agg_logp = np.vstack(agg_logp)
    m = np.max(agg_logp, axis=1, keepdims=True)
    ex = np.exp(agg_logp - m)
    agg_probs = ex / ex.sum(axis=1, keepdims=True)
    return agg_probs, np.array(agg_labels)


def align_by_label_after_aggregation(
    prior_probs: np.ndarray, prior_labels: np.ndarray,
    posterior_probs: np.ndarray, posterior_labels: np.ndarray
):
    """对齐两个数据集"""
    common = np.intersect1d(np.unique(prior_labels), np.unique(posterior_labels))
    prior_idx = {lb: i for i, lb in enumerate(prior_labels)}
    post_idx = {lb: i for i, lb in enumerate(posterior_labels)}
    A, B, L = [], [], []
    for lb in sorted(common):
        A.append(prior_probs[prior_idx[lb]])
        B.append(posterior_probs[post_idx[lb]])
        L.append(lb)
    return np.vstack(A), np.vstack(B), np.array(L)


# ===================== PL 计算 =====================

def calculate_per_sample_pl_variants(
    prior_probs: np.ndarray,
    posterior_probs: np.ndarray,
    labels: np.ndarray,
    class_metric: np.ndarray = None,
    epsilon: float = 1e-12
) -> Dict[int, Dict[str, float]]:
    """计算每个 sample 的多种 PL 指标"""
    N, C = prior_probs.shape
    A = np.clip(prior_probs, epsilon, 1.0)
    B = np.clip(posterior_probs, epsilon, 1.0)
    A = A / A.sum(axis=1, keepdims=True)
    B = B / B.sum(axis=1, keepdims=True)
    
    if class_metric is None:
        D = np.ones((C, C), dtype=float)
        np.fill_diagonal(D, np.inf)
    else:
        D = np.array(class_metric, dtype=float)
        D = np.where(D <= 0, epsilon, D)
    
    per_sample_stats = {}
    for s in range(N):
        la = np.log(A[s])
        lb = np.log(B[s])
        diff = np.abs((lb[:, None] - lb[None, :]) - (la[:, None] - la[None, :]))
        mask = np.triu(np.ones((C, C), dtype=bool), k=1)
        normed = diff[mask] / D[mask]
        per_sample_stats[int(labels[s])] = {
            'mean': float(np.mean(normed)),
            'max': float(np.max(normed)),
            'median': float(np.median(normed)),
            'std': float(np.std(normed)),
            'percentile_95': float(np.percentile(normed, 95)),
            'min': float(np.min(normed))
        }
    return per_sample_stats


# ===================== 可视化（仅保留两个） =====================

def plot_multi_metric_comparison(
    pii_counts: Dict[str, int],
    per_sample_stats: Dict[int, Dict[str, float]],
    label_to_name: Dict[int, str],
    save_dir: str = "pii_leakage_analysis"
):
    """绘制 PIInum vs 多种 PL 指标的 2x2 对比图"""
    x_data = []
    y_mean, y_max, y_median, y_p95 = [], [], [], []
    
    for label, stats in per_sample_stats.items():
        name = label_to_name.get(label, f"Unknown_{label}")
        if name in pii_counts:
            x_data.append(pii_counts[name])
            y_mean.append(stats['mean'])
            y_max.append(stats['max'])
            y_median.append(stats['median'])
            y_p95.append(stats['percentile_95'])
    
    if len(x_data) == 0:
        print("⚠️ 没有匹配的数据可以绘制")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = [
        ('mean', y_mean, 'Mean PL', 'coolwarm'),
        ('max', y_max, 'Max PL (Worst Case)', 'Reds'),
        ('median', y_median, 'Median PL', 'viridis'),
        ('percentile_95', y_p95, '95th Percentile PL', 'plasma')
    ]
    
    for idx, (metric_name, y_data, title, cmap) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        scatter = ax.scatter(x_data, y_data, alpha=0.6, s=100, 
                           c=y_data, cmap=cmap, 
                           edgecolors='black', linewidth=0.5)
        
        z = None
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(x_data), max(x_data), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

        corr = np.corrcoef(x_data, y_data)[0, 1] if len(x_data) > 1 else float("nan")
        ax.set_xlabel('PIInum', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        if z is not None:
            ax.set_title(f'{title}\nCorr={corr:.4f}, y={z[0]:.4f}x+{z[1]:.2f}', fontsize=13)
        else:
            ax.set_title(f'{title}\nCorr=nan', fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "pii_vs_leakage_multi_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 多指标对比图已保存: {save_path}")


def plot_grouped_analysis(
    pii_counts: Dict[str, int],
    per_sample_stats: Dict[int, Dict[str, float]],
    label_to_name: Dict[int, str],
    save_dir: str = "pii_leakage_analysis"
):
    """按 PIInum 分组的箱线图分析"""
    data = []
    for label, stats in per_sample_stats.items():
        name = label_to_name.get(label, f"Unknown_{label}")
        if name in pii_counts:
            data.append({
                'PIInum': pii_counts[name],
                'mean_PL': stats['mean'],
                'max_PL': stats['max'],
                'median_PL': stats['median'],
                'p95_PL': stats['percentile_95']
            })
    
    if len(data) == 0:
        print("⚠️ 没有匹配的数据可以绘制")
        return
    
    df = pd.DataFrame(data)
    df['PII_group'] = pd.cut(df['PIInum'], bins=3, labels=['Low', 'Medium', 'High'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    metrics = [
        ('mean_PL', 'Mean PL'),
        ('max_PL', 'Max PL'),
        ('median_PL', 'Median PL'),
        ('p95_PL', '95th Percentile PL')
    ]
    
    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        df.boxplot(column=col, by='PII_group', ax=ax)
        ax.set_title(f'{title} by PII Group')
        ax.set_xlabel('PII Group')
        ax.set_ylabel(title)
        ax.get_figure().suptitle('')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "pii_grouped_boxplot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 分组箱线图已保存: {save_path}")


# ===================== 主函数 =====================

def comprehensive_analysis(
    data_file: str,
    prior_prob_file: str,
    posterior_prob_file: str,
    class_metric_file: str = "class_distance_matrix.json",
    output_dir: str = "pii_leakage_analysis",
    use_distance_matrix: bool = False,
    label_mapping_file: str = None,
):
    """综合分析主函数"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PIInum vs Posterior Leakage 分析")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    pii_counts = load_pii_counts(data_file)
    
    with open(prior_prob_file, 'r', encoding='utf-8') as f:
        prior_data = json.load(f)
    with open(posterior_prob_file, 'r', encoding='utf-8') as f:
        posterior_data = json.load(f)
    
    # 2. 聚合
    print("\n2. 聚合为对象级概率...")
    if 'logits' in prior_data and 'logits' in posterior_data:
        prior_probs, prior_labels = aggregate_by_label_logits(
            np.array(prior_data['logits']), np.array(prior_data['labels']))
        posterior_probs, posterior_labels = aggregate_by_label_logits(
            np.array(posterior_data['logits']), np.array(posterior_data['labels']))
    else:
        prior_probs, prior_labels = aggregate_by_label_probs_mult(
            np.array(prior_data['probs']), np.array(prior_data['labels']))
        posterior_probs, posterior_labels = aggregate_by_label_probs_mult(
            np.array(posterior_data['probs']), np.array(posterior_data['labels']))
    
    # 3. 对齐
    prior_probs, posterior_probs, labels = align_by_label_after_aggregation(
        prior_probs, prior_labels, posterior_probs, posterior_labels)
    print(f"   对齐后样本数: {len(labels)}")
    
    # 4. 距离矩阵
    class_metric = None
    if use_distance_matrix:
        try:
            with open(class_metric_file, 'r', encoding='utf-8') as f:
                distance_data = json.load(f)
            class_metric = np.array(distance_data['distance_matrix'])
            print(f"   ✅ 已启用距离矩阵: {class_metric.shape}")
        except Exception as e:
            print(f"   ⚠️ 距离矩阵加载失败: {e}")

    # 5. 计算 PL
    print("\n3. 计算后验泄露指标...")
    per_sample_stats = calculate_per_sample_pl_variants(
        prior_probs, posterior_probs, labels, class_metric)

    # 6. 构建 label_to_name
    label_to_name = None
    if label_mapping_file and os.path.isfile(label_mapping_file):
        try:
            label_to_name = load_label_mapping(label_mapping_file)
            print(f"   ✅ 已加载 label_mapping: {len(label_to_name)} 个类别")
        except Exception as e:
            print(f"   ⚠️ 加载 label_mapping 失败: {e}")

    if label_to_name is None:
        df = pd.read_json(data_file)
        name_col = 'Name' if 'Name' in df.columns else 'name'
        if name_col in df.columns:
            unique_names = sorted(df[name_col].unique())
            label_to_name = {i: name for i, name in enumerate(unique_names)}

    # 7. 绘制图表（仅两个）
    print("\n4. 生成可视化...")
    plot_multi_metric_comparison(pii_counts, per_sample_stats, label_to_name, output_dir)
    plot_grouped_analysis(pii_counts, per_sample_stats, label_to_name, output_dir)
    
    # 8. 保存数据
    print("\n5. 保存分析数据...")
    results = []
    for label, stats in per_sample_stats.items():
        name = label_to_name.get(label)
        if name and name in pii_counts:
            results.append({
                'Name': name,
                'PIInum': pii_counts[name],
                'mean_PL': stats['mean'],
                'max_PL': stats['max'],
                'median_PL': stats['median'],
                'p95_PL': stats['percentile_95'],
            })

    df_results = pd.DataFrame(results).sort_values('max_PL', ascending=False)
    csv_path = os.path.join(output_dir, "comprehensive_analysis.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"   ✅ 已保存: {csv_path}")
    
    print("\n" + "=" * 60)
    print("✅ 分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    data_file = r"C:\Users\phdwf\OneDrive\Desktop\textreidentify\TextReIdentification\data\multi_8_all.json"
    prior_prob_file = "outputs/WikiActors/multi_8_onlynoise/eval_results_noise_0p001_abstract.json"
    posterior_prob_file = "outputs/WikiActors/multi_8_onlynoise/eval_results_noise_3p0_abstract.json"

    comprehensive_analysis(
        data_file=data_file,
        prior_prob_file=prior_prob_file,
        posterior_prob_file=posterior_prob_file,
        class_metric_file="distance_matrix_avg.json",
        output_dir="pii_leakage_analysis",
        use_distance_matrix=True,
        label_mapping_file=r"C:\Users\phdwf\OneDrive\Desktop\textreidentify\TextReIdentification\label_mapping.json",
    )