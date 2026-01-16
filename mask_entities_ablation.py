"""
Ablation study script:
Use spaCy NER to find entities and replace them with [MASK] instead of adding noise.
Processes the same two columns: 'Original' and 'Background knowledge'.
"""

import json
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoTokenizer


# ==================== 配置区域 ====================
# ✅ 如果设为 None 或空集合，则对所有 NER 识别的实体进行 MASK
# 如果只想对特定类型处理，可以指定，例如：{"DATE", "PERSON", "ORG", "WORK_OF_ART", "CARDINAL"}
NOISE_LABELS = None  # None 表示对所有 NER 实体进行 MASK

RANDOM_SEED = 42
MODEL_NAME = "distilbert-base-uncased"

DATA_FILE = r"C:\Users\phdwf\OneDrive\Desktop\textreidentify\TextReIdentification\data\multi_8.json"

# 输出文件名带 MASK
OUTPUT_FILE = r"C:\Users\phdwf\OneDrive\Desktop\textreidentify\TextReIdentification\data\multi_8_all.json"

# 你当前数据列名
INPUT_ORIGINAL_COL = "Original"
INPUT_BK_COL = "Background knowledge"


# 是否按实体的 wordpiece 数量生成等量 [MASK]（更可比）
MASK_PER_WORDPIECE = True
REDACTED_TOKEN = "[REDACTED]"


# ==================== 初始化 ====================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_models():
    print("正在加载 spaCy 模型...")
    nlp = spacy.load("en_core_web_lg")

    print(f"正在加载 {MODEL_NAME} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    mask_token = tokenizer.mask_token or "[MASK]"
    print(f"使用 MASK token: {mask_token}")
    return nlp, tokenizer, mask_token


# ==================== MASK 替换逻辑 ====================
def _coerce_text(x) -> str:
    if x is None or pd.isna(x):
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def mask_entities_in_text(text: str, nlp, tokenizer, replace_token: str, use_wordpiece: bool = True) -> Tuple[str, int, int]:
    """
    Returns:
        masked_text
        total_entity_count
        unique_entity_count
    """
    text = _coerce_text(text)
    if text == "":
        return "", 0, 0

    doc = nlp(text)
    # 如果 NOISE_LABELS 为 None 或空，则选择所有实体；否则只选择指定标签的实体
    if NOISE_LABELS:
        selected = [ent for ent in doc.ents if ent.label_ in NOISE_LABELS]
    else:
        selected = list(doc.ents)
    total_count = len(selected)
    unique_count = len(set(ent.text for ent in selected))

    new_text = text
    # 从后往前替换，避免 start/end 偏移
    for ent in sorted(selected, key=lambda e: e.start_char, reverse=True):
        if use_wordpiece and MASK_PER_WORDPIECE:
            n = max(1, len(tokenizer.tokenize(ent.text)))
            repl = " ".join([replace_token] * n)
        else:
            repl = replace_token
        new_text = new_text[:ent.start_char] + repl + new_text[ent.end_char:]

    return new_text, total_count, unique_count


def process_row(row, nlp, tokenizer, mask_token: str):
    # MASK 版本
    masked_abs, total_count, unique_count = mask_entities_in_text(
        row.get(INPUT_ORIGINAL_COL), nlp, tokenizer, mask_token, use_wordpiece=True
    )
    masked_bk, _, _ = mask_entities_in_text(
        row.get(INPUT_BK_COL), nlp, tokenizer, mask_token, use_wordpiece=True
    )
    # REDACTED 版本
    replaced_abs, _, _ = mask_entities_in_text(
        row.get(INPUT_ORIGINAL_COL), nlp, tokenizer, REDACTED_TOKEN, use_wordpiece=False
    )
    replaced_bk, _, _ = mask_entities_in_text(
        row.get(INPUT_BK_COL), nlp, tokenizer, REDACTED_TOKEN, use_wordpiece=False
    )
    return masked_abs, masked_bk, replaced_abs, replaced_bk, total_count, unique_count


def main():
    set_seed(RANDOM_SEED)
    nlp, tokenizer, mask_token = initialize_models()

    print(f"\n加载数据: {DATA_FILE}")
    df = pd.read_json(DATA_FILE)

    # 关键列先清洗，避免 None/NaN 进入 spaCy
    for col in (INPUT_ORIGINAL_COL, INPUT_BK_COL):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    print("开始处理数据（MASK 和 REDACTED 替换实体）...")
    df[["mask_abstract", "mask_public_knowledge", "replaced_abstract", "replaced_background", "entity_count", "unique_entity_count"]] = df.apply(
        lambda row: process_row(row, nlp, tokenizer, mask_token),
        axis=1,
        result_type="expand",
    )

    print(f"\n保存结果到: {OUTPUT_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    save_df = df.drop(columns=["entity_count", "unique_entity_count"], errors="ignore")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(save_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print("\n✅ MASK ablation 完成!")


if __name__ == "__main__":
    main()
