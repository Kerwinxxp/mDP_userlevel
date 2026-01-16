"""
实体噪声添加工具：Original 列支持多组 epsilon budget 加噪（输出到同一文件的多列），
Background knowledge 列使用固定 budget 加噪。

输出：只保留你需要的列（如 Name / Background knowledge / Original / noise_public_knowledge / noise_i_abstract...），
并在每条记录中保存 Original budgets 与 noise_i_abstract 的显式映射 original_budget_map。
"""
from __future__ import annotations

# ✅ 解决 OpenMP 重复加载问题（必须在其他 import 之前）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
from collections import defaultdict

# ==================== 配置区域 ====================
# ✅ 如果设为 None 或空集合，则对所有 NER 识别的实体加噪
# 如果只想对特定类型加噪，可以指定，例如：{"DATE", "PERSON", "ORG", "WORK_OF_ART", "CARDINAL"}
NOISE_LABELS = None  # None 表示对所有 NER 实体加噪

# ✅ Original 多组预算：会生成 noise_1_abstract, noise_2_abstract, ...
ORIGINAL_BUDGETS: List[float] = [0.001, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# ✅ 多个随机种子（多次实验）
RANDOM_SEEDS: List[int] = [42, 7, 15, 11, 36, 49, 2, 5, 8, 19]

# ✅ 是否使用 Budget split（默认 False，使用原始 budget）
USE_BUDGET_SPLIT: bool = False

# ✅ Budget split 公式: new_budget = sqrt(original_budget² / 18) = original_budget / sqrt(18)
# 只对第二个值开始应用 split，第一个值（0.001）保持不变
ORIGINAL_BUDGETS_SPLIT: List[float] = [
    ORIGINAL_BUDGETS[0] if i == 0 else np.sqrt(b ** 2 / 18) 
    for i, b in enumerate(ORIGINAL_BUDGETS)
]

# ✅ Background knowledge 固定预算（只生成一次）- 保持不变
BK_BUDGET: float = 7.0

BUDGET_ALLOCATION_STRATEGY = "independent"  # "shared" or "independent"
CANDIDATE_POOL_SIZE = 50
RANDOM_SEED = 42
MODEL_NAME = "distilbert-base-uncased"

DATA_FILE = r"C:\Users\phdwf\OneDrive\Desktop\textreidentify\TextReIdentification\data\Wiki553_BK=Public+Original.json"

# 输出文件名：与 seed 对齐
OUTPUT_FILE_TEMPLATE = (
    r"C:\Users\phdwf\OneDrive\Desktop\textreidentify\TextReIdentification\data"
    r"\multi_7_seed_{seed}.json"
)

INPUT_ORIGINAL_COL = "Original"
INPUT_BK_COL = "Background knowledge"

# 你常用的“标识列”（如果数据里没有该列会自动跳过）
ID_COLUMNS_CANDIDATES = ["Name"]

# 调试：只在第 1 行、Original 的第 1 个 budget 打印一次 token 分布
DEBUG_PRINT_FIRST_TOKEN_DISTRIBUTION = False
DEBUG_PRINT_NUM_TOKENS = 3
DEBUG_PRINT_TOPK = 20

# ✅ 距离统计配置
ENABLE_DISTANCE_STATS = True
DISTANCE_PLOT_OUTPUT_DIR = r"C:\Users\phdwf\OneDrive\Desktop\textreidentify\TextReIdentification\distance_plots"

# PIInum：仍按你的 scope/unit 统计（统计不随预算变化，因为实体 spans 不变）
PII_COUNT_SCOPE = "original"  # "original" | "bk" | "both"
PII_COUNT_UNIT = "word"  # "entity_span" | "word" | "wp_token"
# ==================== 配置区域结束 ====================


def _fmt_budget(x: float) -> str:
    return str(float(x)).replace(".", "p")


# NEW: noise column name must correspond 1-to-1 with budget value (not index)
def _noise_col_for_budget(b: float) -> str:
    return f"noise_{_fmt_budget(float(b))}_abstract"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_models():
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_lg")

    print(f"Loading transformer model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)
    model.eval()
    return nlp, tokenizer, model


class DistanceStatistics:
    """收集和分析 embedding 距离统计信息"""

    def __init__(self):
        self.all_distances: List[float] = []  # 所有候选 token 的距离
        self.selected_distances: List[float] = []  # 被选中 token 的距离
        self.original_token_distances: List[float] = []  # 原始 token 到自身的距离（应为0或很小）
        self.min_distances: List[float] = []  # 每次选择时的最小距离
        self.max_distances: List[float] = []  # 每次选择时的最大距离
        self.mean_distances: List[float] = []  # 每次选择时候选池的平均距离
        self.distances_by_epsilon: Dict[float, List[float]] = defaultdict(list)  # 按 epsilon 分组的选中距离

    def record(
        self,
        candidate_distances: np.ndarray,
        selected_idx: int,
        epsilon: float,
    ):
        """记录一次 token 替换的距离信息"""
        self.all_distances.extend(candidate_distances.tolist())
        self.selected_distances.append(float(candidate_distances[selected_idx]))
        self.min_distances.append(float(np.min(candidate_distances)))
        self.max_distances.append(float(np.max(candidate_distances)))
        self.mean_distances.append(float(np.mean(candidate_distances)))
        self.distances_by_epsilon[epsilon].append(float(candidate_distances[selected_idx]))

    def print_summary(self):
        """打印距离统计摘要"""
        print("\n" + "=" * 60)
        print("Distance Statistics Summary")
        print("=" * 60)

        if not self.selected_distances:
            print("No distance data collected.")
            return

        selected = np.array(self.selected_distances)
        print(f"\n[Selected Token Distances]")
        print(f"  Count      : {len(selected)}")
        print(f"  Mean       : {np.mean(selected):.6f}")
        print(f"  Std        : {np.std(selected):.6f}")
        print(f"  Min        : {np.min(selected):.6f}")
        print(f"  Max        : {np.max(selected):.6f}")
        print(f"  Median     : {np.median(selected):.6f}")
        print(f"  25th pctl  : {np.percentile(selected, 25):.6f}")
        print(f"  75th pctl  : {np.percentile(selected, 75):.6f}")

        print(f"\n[Candidate Pool Statistics (per selection)]")
        print(f"  Avg min distance : {np.mean(self.min_distances):.6f}")
        print(f"  Avg max distance : {np.mean(self.max_distances):.6f}")
        print(f"  Avg mean distance: {np.mean(self.mean_distances):.6f}")

        print(f"\n[Selected Distances by Epsilon]")
        for eps in sorted(self.distances_by_epsilon.keys()):
            dists = self.distances_by_epsilon[eps]
            print(f"  ε={eps:<6.3f}: n={len(dists):>5}, mean={np.mean(dists):.4f}, std={np.std(dists):.4f}")

    def plot_distributions(self, output_dir: str):
        """生成距离分布图"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.selected_distances:
            print("No distance data to plot.")
            return

        # 1. 选中 token 距离的直方图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        ax1.hist(self.selected_distances, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Distance', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Selected Token Distances', fontsize=14)
        ax1.axvline(np.mean(self.selected_distances), color='red', linestyle='--', label=f'Mean: {np.mean(self.selected_distances):.4f}')
        ax1.axvline(np.median(self.selected_distances), color='green', linestyle='--', label=f'Median: {np.median(self.selected_distances):.4f}')
        ax1.legend()

        # 2. 所有候选 token 距离的直方图
        ax2 = axes[0, 1]
        ax2.hist(self.all_distances, bins=50, edgecolor='black', alpha=0.7, color='coral')
        ax2.set_xlabel('Distance', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of All Candidate Distances', fontsize=14)

        # 3. 按 epsilon 分组的 boxplot
        ax3 = axes[1, 0]
        eps_sorted = sorted(self.distances_by_epsilon.keys())
        data_for_box = [self.distances_by_epsilon[eps] for eps in eps_sorted]
        if data_for_box:
            bp = ax3.boxplot(data_for_box, labels=[f'{eps:.2f}' for eps in eps_sorted], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax3.set_xlabel('Epsilon', fontsize=12)
            ax3.set_ylabel('Selected Distance', fontsize=12)
            ax3.set_title('Selected Distance Distribution by Epsilon', fontsize=14)
            ax3.tick_params(axis='x', rotation=45)

        # 4. epsilon vs 平均选中距离
        ax4 = axes[1, 1]
        mean_by_eps = [(eps, np.mean(self.distances_by_epsilon[eps])) for eps in eps_sorted]
        std_by_eps = [(eps, np.std(self.distances_by_epsilon[eps])) for eps in eps_sorted]
        if mean_by_eps:
            eps_vals = [x[0] for x in mean_by_eps]
            mean_vals = [x[1] for x in mean_by_eps]
            std_vals = [x[1] for x in std_by_eps]
            ax4.errorbar(eps_vals, mean_vals, yerr=std_vals, marker='o', capsize=3, color='darkgreen')
            ax4.set_xlabel('Epsilon', fontsize=12)
            ax4.set_ylabel('Mean Selected Distance', fontsize=12)
            ax4.set_title('Mean Selected Distance vs Epsilon', fontsize=14)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = output_path / "distance_distribution_overview.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved: {fig_path}")

        # 5. 单独的详细直方图（选中距离）
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.selected_distances, bins=100, edgecolor='black', alpha=0.7, color='steelblue', density=True)
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Selected Token Distance Distribution (Normalized)', fontsize=14)

        # 添加统计信息文本框
        stats_text = (
            f"n = {len(self.selected_distances)}\n"
            f"mean = {np.mean(self.selected_distances):.4f}\n"
            f"std = {np.std(self.selected_distances):.4f}\n"
            f"median = {np.median(self.selected_distances):.4f}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig2_path = output_path / "selected_distance_histogram.png"
        plt.savefig(fig2_path, dpi=150)
        plt.close()
        print(f"Saved: {fig2_path}")

        # 6. CDF 图
        fig3, ax = plt.subplots(figsize=(10, 6))
        sorted_dists = np.sort(self.selected_distances)
        cdf = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)
        ax.plot(sorted_dists, cdf, color='navy', linewidth=2)
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('CDF of Selected Token Distances', fontsize=14)
        ax.grid(True, alpha=0.3)

        fig3_path = output_path / "selected_distance_cdf.png"
        plt.savefig(fig3_path, dpi=150)
        plt.close()
        print(f"Saved: {fig3_path}")


# 全局统计收集器
distance_stats = DistanceStatistics()


class ExponentialMechanism:
    """Top-K 最近邻 + 指数机制"""

    def __init__(self, model, tokenizer, candidate_pool_size: int = 50):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = int(tokenizer.vocab_size)
        self.candidate_pool_size = int(min(candidate_pool_size, self.vocab_size))
        self.device = model.device

        print("Preloading vocab embeddings...")
        with torch.no_grad():
            all_token_ids = torch.arange(self.vocab_size, device=self.device)
            self.all_embeddings = model.get_input_embeddings()(all_token_ids)

    def get_token_embedding(self, token: str) -> torch.Tensor:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0 or token_id >= self.vocab_size:
            token_id = self.tokenizer.unk_token_id
        return self.all_embeddings[token_id]

    def select_noisy_token(self, original_embedding: torch.Tensor, epsilon: float):
        distances = torch.norm(self.all_embeddings - original_embedding, p=2, dim=1)
        topk_distances, topk_indices = torch.topk(
            distances, self.candidate_pool_size, largest=False
        )

        candidate_ids = topk_indices.detach().cpu().tolist()
        candidate_distances = topk_distances.detach().cpu().numpy()
        candidate_tokens = self.tokenizer.convert_ids_to_tokens(candidate_ids)

        scaled_scores = -float(epsilon) * candidate_distances / 2.0
        exp_values = np.exp(scaled_scores - np.max(scaled_scores))
        probabilities = exp_values / np.sum(exp_values)

        selected_idx = np.random.choice(len(candidate_ids), p=probabilities)

        # ✅ 记录距离统计
        if ENABLE_DISTANCE_STATS:
            distance_stats.record(candidate_distances, selected_idx, epsilon)

        return candidate_tokens[selected_idx], candidate_tokens, probabilities


_WORD_RE = re.compile(r"\S+")


def _normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if pd.isna(x):
        return ""
    return x if isinstance(x, str) else str(x)


def _extract_entities(text: str, nlp) -> List[Any]:
    if not text:
        return []
    doc = nlp(text)
    # 如果 NOISE_LABELS 为 None 或空，则返回所有实体；否则只返回指定标签的实体
    if NOISE_LABELS:
        return [ent for ent in doc.ents if ent.label_ in NOISE_LABELS]
    else:
        return list(doc.ents)


def _count_units_for_entities(entity_texts: List[str], tokenizer) -> Tuple[int, int, int]:
    span_count = len(entity_texts)
    word_count = sum(len(_WORD_RE.findall(t)) for t in entity_texts)
    wp_token_count = sum(len(tokenizer.tokenize(t)) for t in entity_texts)
    return span_count, word_count, wp_token_count


def _select_unit_count(span_c: int, word_c: int, wp_tok_c: int) -> int:
    if PII_COUNT_UNIT == "word":
        return int(word_c)
    if PII_COUNT_UNIT == "wp_token":
        return int(wp_tok_c)
    return int(span_c)  # "entity_span"


def calculate_epsilon(total_entities: int, strategy: str, total_budget: float) -> float:
    if strategy == "shared":
        return float(total_budget) / total_entities if total_entities > 0 else float(
            total_budget
        )
    if strategy == "independent":
        return float(total_budget)
    raise ValueError(f"Unknown strategy: {strategy}")


def add_noise_to_entity(
    entity_text: str,
    epsilon: float,
    mechanism: ExponentialMechanism,
    tokenizer,
    verbose: bool,
) -> str:
    tokens = tokenizer.tokenize(entity_text)
    noisy_tokens: List[str] = []

    for token_idx, token in enumerate(tokens):
        token_embedding = mechanism.get_token_embedding(token)
        noisy_token, candidate_tokens, probabilities = mechanism.select_noisy_token(
            token_embedding, epsilon
        )
        noisy_tokens.append(noisy_token)

        if verbose and token_idx < DEBUG_PRINT_NUM_TOKENS:
            ranked = sorted(
                zip(candidate_tokens, probabilities), key=lambda x: x[1], reverse=True
            )
            print(f"  Token[{token_idx}] Top-{DEBUG_PRINT_TOPK} candidates for '{token}':")
            for rank, (cand, prob) in enumerate(ranked[:DEBUG_PRINT_TOPK], 1):
                print(f"    {rank:02d}. {cand:>12s}  {prob:.6f}")

    return tokenizer.convert_tokens_to_string(noisy_tokens)


def add_noise_with_entities(
    text: str,
    entities: List[Any],
    epsilon: float,
    mechanism: ExponentialMechanism,
    tokenizer,
    debug_first_entity: bool = False,
) -> str:
    if not text or not entities:
        return text

    new_text = text
    printed = False

    # reverse replace to keep offsets valid
    for ent in sorted(entities, key=lambda e: e.start_char, reverse=True):
        verbose = bool(debug_first_entity and not printed)
        noisy_ent = add_noise_to_entity(
            ent.text, epsilon, mechanism, tokenizer, verbose=verbose
        )
        if verbose:
            printed = True
        new_text = new_text[: ent.start_char] + noisy_ent + new_text[ent.end_char :]

    return new_text


def process_row_multi(
    row: pd.Series,
    nlp,
    mechanism: ExponentialMechanism,
    tokenizer,
    original_budgets: List[float],
    original_budgets_split: List[float],
    bk_budget: float,
    use_split: bool,  # 新增参数
) -> pd.Series:
    original_text = _normalize_text(row.get(INPUT_ORIGINAL_COL))
    bk_text = _normalize_text(row.get(INPUT_BK_COL))

    ents_o = _extract_entities(original_text, nlp)
    ents_b = _extract_entities(bk_text, nlp)

    o_texts = [e.text for e in ents_o]
    b_texts = [e.text for e in ents_b]

    o_span, o_word, o_wp = _count_units_for_entities(o_texts, tokenizer)
    b_span, b_word, b_wp = _count_units_for_entities(b_texts, tokenizer)

    o_unit = _select_unit_count(o_span, o_word, o_wp)
    b_unit = _select_unit_count(b_span, b_word, b_wp)

    if PII_COUNT_SCOPE == "both":
        pii_num = o_unit + b_unit
    elif PII_COUNT_SCOPE == "bk":
        pii_num = b_unit
    else:
        pii_num = o_unit

    out: Dict[str, Any] = {}
    out["PIInum"] = int(pii_num)
    out["PIIunit"] = PII_COUNT_UNIT

    # ✅ 显式保存 Original budgets 与列名的对应关系
    if use_split:
        out["original_budget_map"] = {
            _noise_col_for_budget(b): {"original": float(b), "split": float(b_split)} 
            for b, b_split in zip(original_budgets, original_budgets_split)
        }
    else:
        out["original_budget_map"] = {
            _noise_col_for_budget(b): float(b) for b in original_budgets
        }

    # BK 固定 budget（一次）- 保持不变
    eps_bk = calculate_epsilon(
        total_entities=len(ents_b),
        strategy=BUDGET_ALLOCATION_STRATEGY,
        total_budget=bk_budget,
    )
    out["noise_public_knowledge"] = add_noise_with_entities(
        bk_text, ents_b, eps_bk, mechanism, tokenizer, debug_first_entity=False
    )

    # Original 多 budget（多次）
    debug_this_row = bool(
        DEBUG_PRINT_FIRST_TOKEN_DISTRIBUTION and getattr(row, "name", None) == 0
    )
    for idx, (b, b_split) in enumerate(zip(original_budgets, original_budgets_split)):
        # ✅ 根据开关选择使用原始 budget 还是 split 后的 budget
        budget_to_use = float(b_split) if use_split else float(b)
        eps_o = calculate_epsilon(
            total_entities=len(ents_o),
            strategy=BUDGET_ALLOCATION_STRATEGY,
            total_budget=budget_to_use,
        )
        col = _noise_col_for_budget(b)  # 列名仍用原始 budget
        out[col] = add_noise_with_entities(
            original_text,
            ents_o,
            eps_o,
            mechanism,
            tokenizer,
            debug_first_entity=bool(debug_this_row and idx == 0),
        )

    return pd.Series(out)


def main() -> None:
    if not ORIGINAL_BUDGETS:
        raise ValueError("ORIGINAL_BUDGETS is empty.")
    if len(RANDOM_SEEDS) > 1 and "--in-place" in os.sys.argv:
        raise ValueError("--in-place cannot be used with multiple seeds.")

    nlp, tokenizer, model = initialize_models()
    mechanism = ExponentialMechanism(model, tokenizer, CANDIDATE_POOL_SIZE)

    p = argparse.ArgumentParser(
        description="Add DP-style noise to NER entities in Original and Background knowledge; output only selected columns."
    )
    p.add_argument("--input", default=DATA_FILE, help="Input JSON file path.")
    p.add_argument(
        "--output",
        default="",
        help="Output JSON file path. If empty, uses default OUTPUT_FILE unless --in-place.",
    )
    p.add_argument("--in-place", action="store_true", help="Overwrite input file.")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # 读取输入
    print(f"\nInput: {in_path}")
    df = pd.read_json(str(in_path))

    # 清洗输入列
    for col in (INPUT_ORIGINAL_COL, INPUT_BK_COL):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    print("\n" + "=" * 60)
    print("Config")
    print("=" * 60)
    print(f"Original budgets      : {ORIGINAL_BUDGETS}")
    print(f"Use budget split      : {USE_BUDGET_SPLIT}")
    if USE_BUDGET_SPLIT:
        print(f"Original budgets split: {[round(b, 6) for b in ORIGINAL_BUDGETS_SPLIT]}")
    print(f"BK budget             : {BK_BUDGET}")
    print(f"Strategy              : {BUDGET_ALLOCATION_STRATEGY}")
    print(f"PII scope/unit        : {PII_COUNT_SCOPE} / {PII_COUNT_UNIT}")
    print(f"Seeds                : {RANDOM_SEEDS}")
    print("=" * 60)

    for seed in RANDOM_SEEDS:
        set_seed(seed)

        # 生成输出列
        extra = df.apply(
            lambda r: process_row_multi(
                r, nlp, mechanism, tokenizer, 
                ORIGINAL_BUDGETS, ORIGINAL_BUDGETS_SPLIT, BK_BUDGET, 
                USE_BUDGET_SPLIT
            ),
            axis=1,
            result_type="expand",
        )
        df_out = pd.concat([df, extra], axis=1)

        # ✅ 只保留你用到的列（按 budget 值生成噪声列名）
        noise_cols = [_noise_col_for_budget(b) for b in ORIGINAL_BUDGETS]
        keep_cols = [
            *ID_COLUMNS_CANDIDATES,
            INPUT_BK_COL,
            INPUT_ORIGINAL_COL,
            "noise_public_knowledge",
            *noise_cols,
            "original_budget_map",
            "PIInum",
            "PIIunit",
        ]
        keep_cols = [c for c in keep_cols if c in df_out.columns]
        df_out = df_out[keep_cols]

        # 输出路径
        if args.in_place:
            out_path = in_path
        elif args.output:
            out_path = Path(args.output.format(seed=seed))
        else:
            out_path = Path(OUTPUT_FILE_TEMPLATE.format(seed=seed))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(df_out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

        print(f"\nSaved: {out_path}")
        print(f"Saved columns: {list(df_out.columns)}")

        # ✅ 打印和绘制距离统计
        if ENABLE_DISTANCE_STATS:
            distance_stats.print_summary()
            distance_stats.plot_distributions(DISTANCE_PLOT_OUTPUT_DIR)


if __name__ == "__main__":
    main()