"""Yambda stage 2: temporal audit, held-out protocol, popularity baseline.

Place at yambda-research/src/stage2_protocol.py and run in PyCharm.
Input: data/processed/events_300.parquet from prepare_sample.py.
Outputs: results/stage2_{activity.csv,protocol.json,validation.json}.
Only train/validation *labels* are inspected. Final test labels remain held out.
Requires: pandas, numpy, pyarrow.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "events_300.parquet"
RESULTS = ROOT / "results"

DAY = 86_400  # timestamps in seconds, quantized to 5-second resolution
DATASET_END = 26_000_000  # official Yambda benchmark LAST_TIMESTAMP
HORIZON_DAYS = 7
HORIZON = HORIZON_DAYS * DAY
TRAIN_DAYS_BEFORE_END = [77, 63, 49, 35]
VALIDATION_DAYS_BEFORE_END = 21
TEST_DAYS_BEFORE_END = 7
POPULAR_HISTORY_DAYS = 30
USER_HISTORY_DAYS = 60
POPULAR_CANDIDATES = 500
USER_CANDIDATES = 100
K = 10


def report_snapshot(positives: pd.DataFrame, t: int, name: str) -> dict:
    # Predictions are made immediately before t; labels are [t, t+HORIZON).
    target = positives.loc[
        positives["timestamp"].ge(t) & positives["timestamp"].lt(t + HORIZON),
        ["uid", "item_id"],
    ].drop_duplicates()
    counts = target.groupby("uid").size()
    row = {
        "name": name,
        "cutoff_timestamp": int(t),
        "prediction_day": round(t / DAY, 3),
        "label_interval": [int(t), int(t + HORIZON)],
        "positive_pairs": int(len(target)),
        "users_with_positive": int(len(counts)),
        "positive_items": int(target["item_id"].nunique()),
        "median_positives_per_active_user": float(counts.median()) if len(counts) else 0.0,
    }
    return row


def top_popular(positives: pd.DataFrame, t: int, window_days: int, limit: int) -> list[int]:
    hist = positives.loc[
        positives["timestamp"].ge(t - window_days * DAY)
        & positives["timestamp"].lt(t),
        "item_id",
    ]
    popularity = hist.value_counts().rename_axis("item_id").reset_index(name="count")
    # Deterministic tie-breaking (no information from the future).
    popularity = popularity.sort_values(["count", "item_id"], ascending=[False, True])
    return [int(v) for v in popularity["item_id"].head(limit)]


def user_recent_items(positives: pd.DataFrame, t: int, window_days: int, limit: int) -> dict[int, list[int]]:
    recent = positives.loc[
        positives["timestamp"].ge(t - window_days * DAY)
        & positives["timestamp"].lt(t),
        ["uid", "item_id", "timestamp"],
    ]
    # Most recent distinct tracks listened to by each user.
    recent = recent.sort_values(
        ["uid", "timestamp", "item_id"], ascending=[True, False, True]
    ).drop_duplicates(["uid", "item_id"], keep="first")
    recent = recent.groupby("uid", sort=False).head(limit)
    return {
        int(uid): [int(item) for item in part["item_id"]]
        for uid, part in recent.groupby("uid", sort=False)
    }


def evaluate_validation(positives: pd.DataFrame, all_events: pd.DataFrame, t: int) -> dict:
    labels = positives.loc[
        positives["timestamp"].ge(t) & positives["timestamp"].lt(t + HORIZON),
        ["uid", "item_id"],
    ].drop_duplicates()
    ground_truth = {
        int(uid): set(int(v) for v in part["item_id"])
        for uid, part in labels.groupby("uid")
    }
    popular = top_popular(positives, t, POPULAR_HISTORY_DAYS, POPULAR_CANDIDATES)
    recent = user_recent_items(positives, t, USER_HISTORY_DAYS, USER_CANDIDATES)
    catalog = set(int(i) for i in all_events.loc[
        all_events["timestamp"].lt(t), "item_id"
    ].unique())

    if not ground_truth:
        raise ValueError("Validation has no positive users: revise protocol before training.")
    if len(popular) < K:
        raise ValueError("Fewer than K popular candidate items before validation cutoff.")

    candidate_micro_hits = 0
    positive_pairs = 0
    macro_candidate_recalls = []
    macro_pop_recalls = []
    macro_ndcgs = []
    candidate_sizes = []
    total_novel_item_pairs = 0

    ranked = popular[:K]
    ranked_set = set(ranked)
    for uid, truth in ground_truth.items():
        candidates = set(popular) | set(recent.get(uid, []))
        hits = len(truth & candidates)
        positive_pairs += len(truth)
        candidate_micro_hits += hits
        macro_candidate_recalls.append(hits / len(truth))
        candidate_sizes.append(len(candidates))
        macro_pop_recalls.append(len(truth & ranked_set) / len(truth))
        dcg = sum(1.0 / math.log2(pos + 2) for pos, item in enumerate(ranked) if item in truth)
        ideal_dcg = sum(1.0 / math.log2(pos + 2) for pos in range(min(K, len(truth))))
        macro_ndcgs.append(dcg / ideal_dcg)
        total_novel_item_pairs += len(truth - catalog)

    metrics = {
        "validation_cutoff": int(t),
        "evaluated_users_with_at_least_one_positive": len(ground_truth),
        "total_distinct_user_item_positives": int(positive_pairs),
        "positive_pairs_not_previously_in_catalog": int(total_novel_item_pairs),
        "new_catalog_item_fraction": float(total_novel_item_pairs / positive_pairs),
        "candidate_generation": f"global top-{POPULAR_CANDIDATES} 30d positives UNION user last-{USER_CANDIDATES} distinct 60d positives",
        "mean_candidates_per_evaluated_user": float(np.mean(candidate_sizes)),
        "candidate_recall_micro": float(candidate_micro_hits / positive_pairs),
        "candidate_recall_macro": float(np.mean(macro_candidate_recalls)),
        "popularity_recall_at_10_macro": float(np.mean(macro_pop_recalls)),
        "popularity_ndcg_at_10_macro": float(np.mean(macro_ndcgs)),
        "ranking_baseline": "same global top-10 positive-listen items for all users; repeat items allowed",
        "important": "These are offline replay metrics on observed future listens, not CTR or causal relevance.",
    }
    return metrics


def main() -> None:
    if not DATA.is_file():
        raise FileNotFoundError(f"Input not found: {DATA}")
    RESULTS.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(DATA, columns=[
        "uid", "item_id", "timestamp", "event_type", "played_ratio_pct", "is_organic"
    ])
    assert len(data) > 0 and not data[["uid", "item_id", "timestamp"]].isna().any().any()
    assert data["timestamp"].ge(0).all() and data["timestamp"].lt(DATASET_END).all(), (
        "Unexpected timestamps: check dataset revision and official benchmark bounds"
    )
    assert data["uid"].nunique() == 300, "Expected exactly 300 selected users"

    remainders = data["timestamp"].mod(5).value_counts().sort_index()
    print("\n=== Timestamp sanity check ===")
    print("Min/max:", int(data["timestamp"].min()), int(data["timestamp"].max()))
    print("Observed span in DAYS (timestamps are seconds):",
          round((int(data["timestamp"].max()) - int(data["timestamp"].min())) / DAY, 2))
    print("timestamp % 5 frequencies:", remainders.to_dict())
    if (data["timestamp"] % 5 != 0).any():
        print("WARNING: some timestamps are not multiples of five. Investigate before modeling!")

    data["positive"] = data["event_type"].eq("listen") & data["played_ratio_pct"].gt(50)
    pos = data.loc[data["positive"], ["uid", "item_id", "timestamp"]].copy()
    # Descriptive activity is restricted to pre-validation events.
    # We do not examine validation/test-period label frequencies to set the protocol.
    val_t = DATASET_END - VALIDATION_DAYS_BEFORE_END * DAY
    pre_validation = data.loc[data["timestamp"].lt(val_t)]
    month = pre_validation["timestamp"] // (30 * DAY)
    activity = pre_validation.groupby(month, sort=True).agg(
        all_events=("uid", "size"),
        positive_listens=("positive", "sum"),
        active_users=("uid", "nunique"),
    ).reset_index().rename(columns={"timestamp": "30_day_bin"})
    activity["bin_start_day"] = activity["30_day_bin"] * 30
    activity.to_csv(RESULTS / "stage2_activity.csv", index=False)
    print("\n=== 30-day activity bins ===")
    print(activity[["bin_start_day", "all_events", "positive_listens", "active_users"]].to_string(index=False))

    train_times = [DATASET_END - d * DAY for d in TRAIN_DAYS_BEFORE_END]
    test_t = DATASET_END - TEST_DAYS_BEFORE_END * DAY
    assert max(train_times) + HORIZON <= val_t
    assert val_t + HORIZON <= test_t
    assert test_t + HORIZON <= DATASET_END

    snapshots = [report_snapshot(pos, t, f"train_{j+1}") for j, t in enumerate(train_times)]
    snapshots.append(report_snapshot(pos, val_t, "validation"))
    # Do not inspect or report test labels before finalizing model selection.
    protocol = {
        "dataset": "Yambda-50M, random full-history 300-user sample seed=42",
        "timestamp_unit": "seconds, 5-second resolution (not five-second bucket indices)",
        "official_dataset_end": DATASET_END,
        "label": "distinct (uid,item_id) with event_type=='listen' and played_ratio_pct>50",
        "prediction": "at cutoff t, features from events timestamp < t, targets [t,t+604800)",
        "horizon_days": HORIZON_DAYS,
        "train_snapshots": snapshots[:-1],
        "validation_snapshot": snapshots[-1],
        "test_prediction_cutoff": test_t,
        "test_labels_evaluated": False,
        "candidate_method": "global 500 most listened 30d + user 100 recent unique 60d",
        "candidate_method_freeze_before_final_test": True,
        "repeat_listens_allowed": True,
    }
    (RESULTS / "stage2_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== Fixed temporal protocol ===")
    for snapshot in snapshots:
        print(f"{snapshot['name']:>12}: cutoff={snapshot['cutoff_timestamp']} "
              f"({snapshot['prediction_day']:.1f}d), "
              f"7d future: users={snapshot['users_with_positive']}, "
              f"distinct positives={snapshot['positive_pairs']}")
    print(f"        test: prediction cutoff={test_t}; TEST LABELS NOT INSPECTED")

    metrics = evaluate_validation(pos, data, val_t)
    (RESULTS / "stage2_validation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== Validation (popularity baseline & candidate diagnostic) ===")
    for key in [
        "evaluated_users_with_at_least_one_positive",
        "total_distinct_user_item_positives",
        "new_catalog_item_fraction",
        "mean_candidates_per_evaluated_user",
        "candidate_recall_micro",
        "candidate_recall_macro",
        "popularity_recall_at_10_macro",
        "popularity_ndcg_at_10_macro",
    ]:
        print(f"{key}: {metrics[key]:.5f}" if isinstance(metrics[key], float)
              else f"{key}: {metrics[key]}")
    print("\nSaved:")
    for name in ["stage2_activity.csv", "stage2_protocol.json", "stage2_validation.json"]:
        print(" -", RESULTS / name)
    print("\nIMPORTANT: Test labels must not be inspected until the final evaluation.")


if __name__ == "__main__":
    main()
