
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "yandex/yambda"

FILES = [
    "sequential/50m/multi_event.parquet",
    "artist_item_mapping.parquet",
    "album_item_mapping.parquet",
]

def main():
    revision = HfApi().dataset_info(REPO_ID).sha

    print("Dataset revision:", revision)

    for filename in FILES:
        print(f"\nDownloading: {filename}")

        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            local_dir=str(DATA_DIR),
        )

        print("Saved to:", path)

    (DATA_DIR / "dataset_revision.txt").write_text(
        revision, encoding="utf-8"
    )

    print("\nDownload complete!")


if __name__ == "__main__":
    main()
