import json
import math
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import spacy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt

# =========================
# 直接运行配置（不走命令行）
DEFAULT_DATA_FILE = r"C:\Users\phdwf\OneDrive\Desktop\textreidentify\TextReIdentification\data\multi.json"
DEFAULT_ID_COL = "Name"
DEFAULT_TEXT_COL = "Original"
DEFAULT_MISSING_PENALTY = 1.5

# =========================
DATA_FILE = DEFAULT_DATA_FILE
ID_COL = DEFAULT_ID_COL
TEXT_COL = DEFAULT_TEXT_COL

OUT_MATRIX_JSON = "distance_matrix_avg.json"
OUT_FIRST_USER_CSV = "first_user_distances.csv"
OUT_PLOT_PNG = "first_user_distance_plot.png"
OUT_ALL_PAIRS_CSV = "all_pair_distances.csv"
OUT_ALL_PAIRS_PNG = "all_pair_distance_plot.png"

ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64
MISSING_PENALTY = DEFAULT_MISSING_PENALTY
MAX_ENTS_PER_LABEL = 0  # >0 可加速：每个label最多保留N个实体
LABEL_UNIVERSE_MODE = "model"  # "model" | "dataset" | "pair"（推荐 model，维度固定）

# ✅ 是否对权重进行归一化（L2 归一化：||w||_2 = 1）
NORMALIZE_WEIGHTS = True

# 你原来的权重，其他 label 默认 1.0（="用所有的"）
DEFAULT_DIMENSION_WEIGHTS = {
    "PERSON": 1.0,
    "ORG": 1.0,
    "DATE": 1.0,
    "CARDINAL": 1.0,
}



def norm(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return x if isinstance(x, str) else str(x)


def load_spacy():
    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        return spacy.load("en_core_web_sm")


def iter_ner_entities(nlp, texts, batch_size=64):
    for doc in nlp.pipe(texts, batch_size=batch_size):
        yield [(ent.text, ent.label_) for ent in doc.ents]


def build_users_from_df(nlp, df, id_col, text_col, batch_size=64):
    ids = [norm(v) for v in df.get(id_col, pd.Series([])).tolist()]
    texts = [norm(v) for v in df.get(text_col, pd.Series([])).tolist()]
    users = []
    for uid, ents in zip(ids, iter_ner_entities(nlp, texts, batch_size=batch_size)):
        users.append({"id": uid, "entities": ents})
    return users


def _pad_to_len(items, target_len):
    if not items:
        return []
    if len(items) == target_len:
        return items
    q, r = divmod(target_len, len(items))
    return items * q + items[:r]


def compute_inner_emd_from_embeddings(emb_a: np.ndarray, emb_b: np.ndarray, missing_penalty: float) -> float:
    """
    Compute an EMD-style distance between two *multisets* of embeddings.

    In the CCS'24 EMD framework, when two empirical datasets have the SAME size m,
    the 1-Wasserstein distance equals the minimum-cost perfect matching (a permutation)
    averaged by m (Birkhoff–von Neumann; Lemma 2.1).

    Your data is generally UNBOUNDED: the number of entities under a label can differ
    across users. For unequal sizes, we follow the same "minimum-cost coupling" intuition
    by allowing unmatched mass to be transported to/from a dummy 'null' element at a
    fixed cost (missing_penalty). Concretely, we:
      - set n = max(na, nb)
      - build an n×n cost matrix:
          * real-real: Euclidean distance between normalized embeddings
          * real-dummy / dummy-real: missing_penalty
          * dummy-dummy: 0
      - solve a minimum-cost perfect matching via Hungarian algorithm
      - return average cost (sum / n)

    This avoids the previous replication-based padding (which can distort transport mass),
    while preserving the bounded-case matching semantics when na==nb.
    """
    na = 0 if emb_a is None else int(len(emb_a))
    nb = 0 if emb_b is None else int(len(emb_b))

    # both empty => identical
    if na == 0 and nb == 0:
        return 0.0

    # one empty => all mass goes to dummy at missing_penalty
    if na == 0 or nb == 0:
        return float(missing_penalty)

    # bounded case (same size) still works as a special case of the dummy construction
    n = max(na, nb)

    # cost matrix
    # Top-left: real-real distances
    d_mat = np.full((n, n), float(missing_penalty), dtype=np.float64)

    # real-real block
    d_real = cdist(emb_a, emb_b, metric="euclidean")
    d_mat[:na, :nb] = d_real

    # dummy-dummy block: 0 cost
    if na < n and nb < n:
        d_mat[na:, nb:] = 0.0
    elif na < n and nb == n:
        # only A has dummies, B is all real -> dummy-real already set to missing_penalty
        pass
    elif nb < n and na == n:
        # only B has dummies, A is all real -> real-dummy already set to missing_penalty
        pass

    row_ind, col_ind = linear_sum_assignment(d_mat)
    return float(d_mat[row_ind, col_ind].sum() / n)



def compute_user_pair_l2_distance(user_a_by_label, user_b_by_label, weights, missing_penalty, label_universe=None):
    """
    核心概念（直接写在代码里，避免误解）：
    - 这里的“user vector”不是一个固定长度的 dense vector。
      而是：按 NER label 分组的“可变长 embedding 集合”：：
        user_by_label[label] = [emb1, emb2, ...]  (可能为空)
    - 所谓“每个维度”，指的是每个 NER label（PERSON/ORG/GPE/...）。
    - 某个 label 在一边有实体、另一边为空时，不做匹配，直接返回 missing_penalty（默认 1.5）。
      所以你看到 raw=1.500000 往往就是缺失惩罚，不是模型算出来的欧式距离。

    参数：
    - label_universe=None：仅在这对样本出现过的 label 上聚合（pair union，维度会随 pair 变化）
    - label_universe=iterable：使用固定的 label 全集（维度固定，便于 debug/对齐）
    """
    if label_universe is None:
        labels = sorted(set(user_a_by_label.keys()) | set(user_b_by_label.keys()))  # 稳定顺序
    else:
        labels = list(label_universe)  # 由外部保证顺序（下面我们会传 sorted(...)）

    weighted_sq_sum = 0.0
    details = {}

    for lab in labels:
        w = float(weights.get(lab, 1.0))  # 不在权重表里的 label，权重默认 1.0
        da = user_a_by_label.get(lab)
        db = user_b_by_label.get(lab)

        dist = compute_inner_emd_from_embeddings(da, db, missing_penalty=missing_penalty)
        weighted_sq_sum += (dist * w) ** 2
        details[lab] = dist

    return float(np.sqrt(weighted_sq_sum)), details


def _format_dim_debug(details, weights, show_zeros=False, eps=1e-12):
    """
    debug 打印说明：
    - raw: 该 label 下的“内部匹配距离”（或缺失惩罚 missing_penalty）
    - weighted: raw * weight
    - 最终距离是对所有 label 的 weighted 做 L2 norm：sqrt(sum(weighted^2))
    """
    rows = []
    for lab, raw in details.items():
        w = float(weights.get(lab, 1.0))
        wd = float(raw) * w
        if (not show_zeros) and abs(wd) <= eps and abs(float(raw)) <= eps:
            continue
        rows.append((abs(wd), lab, float(raw), w, wd))

    rows.sort(reverse=True)

    lines = []
    for _, lab, raw, w, wd in rows:
        lines.append(f"    {lab:12s} raw={raw:.6f}  w={w:.3f}  weighted={wd:.6f}")
    if not lines:
        return "    (all zero)"
    return "\n".join(lines)


def plot_first_user_distances(dist_vec, users, out_png):
    """
    dist_vec: shape [N]，dist_vec[0]=0
    users: list[str] user ids
    """
    vals = np.asarray(dist_vec[1:], dtype=float)
    if len(vals) == 0:
        print("No other users to plot.")
        return

    # 基本统计
    stats = {
        "count": int(len(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
    }
    print("\n[First-user distance stats]")
    for k, v in stats.items():
        print(f"  {k:>6s}: {v:.6f}" if isinstance(v, float) else f"  {k:>6s}: {v}")

    # 图：左=直方图，右=排序曲线
    sorted_vals = np.sort(vals)

    plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(vals, bins=30, color="steelblue", alpha=0.85)
    ax1.set_title("Distances from User[0] to others (Histogram)")
    ax1.set_xlabel("distance")
    ax1.set_ylabel("count")

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(sorted_vals, linewidth=1.5, color="darkorange")
    ax2.set_title("Distances from User[0] to others (Sorted)")
    ax2.set_xlabel("rank (sorted)")
    ax2.set_ylabel("distance")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print("\nSaved plot:", out_png)


def plot_all_pair_distances(dist_mat: np.ndarray, out_png: str, out_csv: Optional[str] = None):
    """
    所有 user 的两两距离（上三角，不含对角）：
      count = N*(N-1)/2
    """
    n = int(dist_mat.shape[0])
    if n <= 1:
        print("Not enough users to plot all-pairs distances.")
        return

    tri = np.triu_indices(n, k=1)
    vals = dist_mat[tri].astype(np.float64, copy=False)
    if vals.size == 0:
        print("No pair distances to plot.")
        return

    stats = {
        "count": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
    }
    print("\n[All-pairs distance stats]  (upper triangle, excluding diagonal)")
    for k, v in stats.items():
        print(f"  {k:>6s}: {v:.6f}" if isinstance(v, float) else f"  {k:>6s}: {v}")

    if out_csv:
        pd.DataFrame({"distance": vals}).to_csv(out_csv, index=False, encoding="utf-8")
        print("Saved:", out_csv)

    sorted_vals = np.sort(vals)

    plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(vals, bins=50, color="seagreen", alpha=0.85)
    ax1.set_title(f"All-pairs distances (Histogram)  count={vals.size}")
    ax1.set_xlabel("distance")
    ax1.set_ylabel("count")

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(sorted_vals, linewidth=1.2, color="purple")
    ax2.set_title("All-pairs distances (Sorted)")
    ax2.set_xlabel("rank (sorted)")
    ax2.set_ylabel("distance")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print("Saved plot:", out_png)


def _stable_sort_df_by_tri_label_order(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    TRI 的 label 顺序来自：sorted(list(all_individuals))，也就是按“姓名字符串”排序。
    这里在距离计算前对 df 做同样的稳定排序，从而 users[0] <-> label 1 (name_to_label=0)。

    参数：
    - df: 输入的 DataFrame
    - id_col: 用于排序的 ID 列名

    返回：
    - 排序后的 DataFrame
    """
    if id_col not in df.columns:
        return df

    df2 = df.copy()
    sort_key = df2[id_col].map(norm).map(lambda s: s.strip())
    df2["_tri_sort_key"] = sort_key

    # 空/缺失 id 放最后，避免把无效行排到最前导致整体错位
    df2["_tri_is_empty"] = df2["_tri_sort_key"].eq("") | df2["_tri_sort_key"].isna()

    # mergesort 保持稳定（同名多行时维持原相对顺序）
    df2 = df2.sort_values(by=["_tri_is_empty", "_tri_sort_key"], kind="mergesort").reset_index(drop=True)

    return df2.drop(columns=["_tri_sort_key", "_tri_is_empty"])


def normalize_weights_l2(weights: dict, label_universe: list) -> dict:
    """
    L2-normalize weights over a fixed label universe (recommended when the outer
    aggregation is an L2 norm).

    We compute, for each label in label_universe:
      w_raw(label) = weights.get(label, 1.0)
    and then return:
      w(label) = w_raw(label) / sqrt(sum_j w_raw(j)^2)

    This keeps the *scale* of the user-level distance stable when you change the
    number of labels or their relative magnitudes (important if the distance is later
    used inside exponentials / softmax, etc.).
    """
    if not label_universe:
        return weights.copy()

    raw = np.array([float(weights.get(lab, 1.0)) for lab in label_universe], dtype=np.float64)
    norm = float(np.linalg.norm(raw, ord=2))
    if norm <= 0:
        # fallback to uniform weights
        uniform = 1.0 / math.sqrt(len(label_universe))
        return {lab: uniform for lab in label_universe}

    return {lab: float(weights.get(lab, 1.0)) / norm for lab in label_universe}



def print_weight_info(weights: dict, normalized_weights: dict, label_universe: list):
    """打印权重信息，便于调试"""
    print("\n[Weight Configuration]")
    print(f"  Normalize weights: {NORMALIZE_WEIGHTS}")
    print(f"  Label universe size: {len(label_universe)}")
    print("\n  Label            Raw Weight    Normalized Weight")
    print("  " + "-" * 50)
    for lab in sorted(label_universe):
        raw_w = float(weights.get(lab, 1.0))
        norm_w = normalized_weights.get(lab, 0.0)
        print(f"  {lab:16s}  {raw_w:>10.4f}    {norm_w:>10.6f}")
    
    total_raw = sum(float(weights.get(lab, 1.0)) for lab in label_universe)
    total_norm = sum(normalized_weights.get(lab, 0.0) for lab in label_universe)
    print("  " + "-" * 50)
    print(f"  {'TOTAL':16s}  {total_raw:>10.4f}    {total_norm:>10.6f}")


def main():
    df = pd.read_json(DATA_FILE)

    # 👇 关键改动：对齐 TRI 的 label 顺序（姓名字符串排序）
    df = _stable_sort_df_by_tri_label_order(df, ID_COL)

    nlp = load_spacy()
    st_model = SentenceTransformer(ST_MODEL_NAME)

    print("Input:", DATA_FILE)
    print("Rows :", len(df))
    # 可选：快速确认排序效果
    if ID_COL in df.columns:
        head_ids = [norm(x).strip() for x in df[ID_COL].head(5).tolist()]
        print("[TRI-aligned user order] first 5 ids:", head_ids)

    users = build_users_from_df(nlp, df, ID_COL, TEXT_COL, batch_size=BATCH_SIZE)

    # label counts（数据里出现过的label）
    label_counts = Counter()
    for u in users:
        for _, lab in u["entities"]:
            label_counts[lab] += 1
    print("\n[NER label counts in dataset]")
    for lab, cnt in label_counts.most_common():
        print(f"  {lab:12s}  {cnt:8d}")

    # label universe（决定"维度全集"）
    if LABEL_UNIVERSE_MODE == "pair":
        label_universe = None
    elif LABEL_UNIVERSE_MODE == "dataset":
        label_universe = sorted(label_counts.keys())
    else:  # "model"
        label_universe = sorted(nlp.get_pipe("ner").labels)

    # ✅ 权重归一化
    weights = dict(DEFAULT_DIMENSION_WEIGHTS)
    if NORMALIZE_WEIGHTS and label_universe is not None:
        normalized_weights = normalize_weights_l2(weights, label_universe)
        print_weight_info(weights, normalized_weights, label_universe)
        weights = normalized_weights
    elif label_universe is not None:
        # 不归一化时也打印信息
        print_weight_info(weights, weights, label_universe)

    # ---- 向量化（按 label 去重缓存）----
    label_to_texts = defaultdict(list)
    seen = defaultdict(set)
    for u in users:
        for t, lab in u["entities"]:
            t = norm(t).strip()
            if not t:
                continue
            if t not in seen[lab]:
                seen[lab].add(t)
                label_to_texts[lab].append(t)

    label_to_emb = {}
    for lab, texts in label_to_texts.items():
        if not texts:
            label_to_emb[lab] = {}
            continue
        embs = st_model.encode(texts, normalize_embeddings=True, batch_size=BATCH_SIZE)
        label_to_emb[lab] = {t: embs[i] for i, t in enumerate(texts)}
        print(f"Embedded label={lab:12s} unique_texts={len(texts)}")

    user_embs = []
    for u in users:
        by_label = defaultdict(list)
        for t, lab in u["entities"]:
            t = norm(t).strip()
            if not t:
                continue
            emb = label_to_emb.get(lab, {}).get(t)
            if emb is not None:
                by_label[lab].append(emb)

        if MAX_ENTS_PER_LABEL and MAX_ENTS_PER_LABEL > 0:
            for lab in list(by_label.keys()):
                by_label[lab] = by_label[lab][:MAX_ENTS_PER_LABEL]

        by_label_np = {}
        for lab, lst in by_label.items():
            if lst:
                by_label_np[lab] = np.vstack(lst)
        user_embs.append(by_label_np)

    # ---- 距离矩阵 + 第1个user到所有人的距离 ----
    N = len(users)
    dist_mat = np.zeros((N, N), dtype=np.float32)
    # weights 已在上面处理过（归一化或保持原样）

    first_user_dist = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        if i % 10 == 0 or i == N - 1:
            print(f"Computing row {i+1}/{N} ...")
        for j in range(i + 1, N):
            d, _ = compute_user_pair_l2_distance(
                user_embs[i],
                user_embs[j],
                weights=weights,
                missing_penalty=MISSING_PENALTY,
                label_universe=label_universe,
            )
            dist_mat[i, j] = d
            dist_mat[j, i] = d
            if i == 0:
                first_user_dist[j] = d

    # 打印第1个user到所有user
    u0 = users[0]["id"] if users else ""
    print("\n" + "=" * 80)
    print(f"[Distances] First user (index 1) = {u0!r}")
    print("index\tuser_id\tdistance")
    for idx in range(N):
        uid = users[idx]["id"]
        print(f"{idx+1}\t{uid}\t{float(first_user_dist[idx]):.6f}")

    # 保存 CSV（第1个user距离向量）
    pd.DataFrame(
        {"index_1based": np.arange(1, N + 1), "user_id": [u["id"] for u in users], "distance": first_user_dist}
    ).to_csv(OUT_FIRST_USER_CSV, index=False, encoding="utf-8")
    print("\nSaved:", OUT_FIRST_USER_CSV)

    # 画图（直方图+排序曲线）
    plot_first_user_distances(first_user_dist, [u["id"] for u in users], OUT_PLOT_PNG)

    # 新增：所有 pairs 的分布图（N*(N-1)/2 个距离）
    plot_all_pair_distances(dist_mat, OUT_ALL_PAIRS_PNG, out_csv=OUT_ALL_PAIRS_CSV)

    # 保存 matrix JSON
    out_obj = {
        "distance_matrix": dist_mat.tolist(),
        "shape": [N, N],
        "metric": "emd_per_label_then_l2",
        "missing_penalty": float(MISSING_PENALTY),
        "weights": weights,
        "sentence_transformer": ST_MODEL_NAME,
        "label_universe_mode": LABEL_UNIVERSE_MODE,
        "users": [u["id"] for u in users],
        "label_counts": dict(label_counts),
        "id_col": ID_COL,
        "text_col": TEXT_COL,
        "data": DATA_FILE,
    }
    with open(OUT_MATRIX_JSON, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False)

    print("\nSaved:", OUT_MATRIX_JSON)
    print("Matrix shape:", out_obj["shape"])


if __name__ == "__main__":
    main()
