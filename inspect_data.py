
from pathlib import Path
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent / "data"

FILES = [
    "sequential/50m/multi_event.parquet",
    "artist_item_mapping.parquet",
    "album_item_mapping.parquet",
]

for filename in FILES:
    path = DATA_DIR / filename

    print(f"\n{'=' * 60}")
    print("File:", filename)

    if not path.exists():
        print("ERROR: File not found")
        continue

    parquet = pq.ParquetFile(path)

    print("Size (MB):", round(path.stat().st_size / 1024**2, 2))
    print("Rows:", parquet.metadata.num_rows)
    print("Schema:")
    print(parquet.schema_arrow)

print("\nInspection complete!")
