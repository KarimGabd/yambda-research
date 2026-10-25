"""Select complete histories of 300 Yambda users and inspect them.

Place at yambda-research/src/prepare_sample.py and run with the project interpreter.
Requires: numpy, pandas, pyarrow.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "sequential" / "50m" / "multi_event.parquet"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT = OUTPUT_DIR / "events_300.parquet"
N_USERS = 300
SEED = 42
EVENT_COLUMNS = (
    "timestamp", "item_id", "is_organic", "played_ratio_pct",
    "track_length_seconds", "event_type",
)


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Input data not found: {SOURCE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # The UID column contains one row per user, not one row per event.
    all_uids = pq.read_table(SOURCE, columns=["uid"])["uid"].to_numpy()
    if len(np.unique(all_uids)) != len(all_uids):
        raise ValueError("Expected exactly one row per UID")
    if N_USERS > len(all_uids):
        raise ValueError("Requested more users than are available")

    # The first 300 users of this permutation are also included in its first 1000.
    frozen_uids = ROOT / "reproducibility" / "selected_uids_300.txt"

    if frozen_uids.exists():
        selected = np.loadtxt(
            frozen_uids, dtype=np.uint32, ndmin=1
        )
        if (
                len(selected) != N_USERS
                or len(np.unique(selected)) != N_USERS
                or not np.isin(selected, all_uids).all()
        ):
            raise ValueError("Invalid frozen user selection")
    else:
        selected = np.random.default_rng(SEED).permutation(
            all_uids
        )[:N_USERS]
    selected_set = pa.array(selected, type=pa.uint32())
    np.savetxt(OUTPUT_DIR / "selected_uids_300.txt", np.sort(selected), fmt="%d")

    writer = None
    selected_rows = 0
    total_events = 0
    parquet = pq.ParquetFile(SOURCE)

    try:
        for batch in parquet.iter_batches(batch_size=64):
            mask = pc.is_in(batch.column(batch.schema.get_field_index("uid")),
                            value_set=selected_set)
            if not pc.any(mask).as_py():
                continue

            picked = batch.filter(mask)
            selected_rows += picked.num_rows
            lists = {
                name: picked.column(picked.schema.get_field_index(name))
                for name in EVENT_COLUMNS
            }
            lengths = pc.list_value_length(lists["timestamp"]).to_numpy()
            for name, values in lists.items():
                if not np.array_equal(lengths, pc.list_value_length(values).to_numpy()):
                    raise ValueError(f"Misaligned event lists: {name}")

            flat = pa.table({
                "uid": pa.array(np.repeat(picked.column(
                    picked.schema.get_field_index("uid")).to_numpy(), lengths),
                    type=pa.uint32()),
                "timestamp": pc.list_flatten(lists["timestamp"]),
                "item_id": pc.list_flatten(lists["item_id"]),
                "is_organic": pc.list_flatten(lists["is_organic"]),
                "played_ratio_pct": pc.list_flatten(lists["played_ratio_pct"]),
                "track_length_seconds": pc.list_flatten(lists["track_length_seconds"]),
                "event_type": pc.cast(pc.list_flatten(lists["event_type"]), pa.string()),
            })

            if writer is None:
                writer = pq.ParquetWriter(OUTPUT, flat.schema, compression="snappy")
            writer.write_table(flat)
            total_events += flat.num_rows
    finally:
        if writer is not None:
            writer.close()

    if selected_rows != N_USERS:
        raise AssertionError(f"Expected {N_USERS} selected users, got {selected_rows}")
    if pq.read_metadata(OUTPUT).num_rows != total_events:
        raise AssertionError("Written event count does not match")

    # Read only the columns needed for first exploratory statistics.
    df = pd.read_parquet(OUTPUT, columns=[
        "uid", "item_id", "timestamp", "is_organic", "event_type", "played_ratio_pct"
    ])
    events_per_user = df.groupby("uid").size()
    t_min = int(df["timestamp"].min())
    t_max = int(df["timestamp"].max())
    complete_listen = ((df["event_type"] == "listen") &
                       (df["played_ratio_pct"] > 50))

    print("\n=== Sample summary ===")
    print("Users:", df["uid"].nunique())
    print("Events:", len(df))
    print("Unique items:", df["item_id"].nunique())
    print("Events/user (min, median, max):",
          int(events_per_user.min()), int(events_per_user.median()),
          int(events_per_user.max()))
    print("Event types:\n", df["event_type"].value_counts(dropna=False).to_string())
    print("Organic flags:\n", df["is_organic"].value_counts(dropna=False).to_string())
    print("Positive listens (played_ratio_pct > 50):", int(complete_listen.sum()))
    print("Timestamp min/max (seconds, 5s precision):", t_min, t_max)
    print("Observed global span (days):", round((t_max - t_min) / 86400, 2))
    print("Sample saved to:", OUTPUT)
    print("UIDs saved to:", OUTPUT_DIR / "selected_uids_300.txt")


if __name__ == "__main__":
    main()
