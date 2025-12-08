import json
import os
import re
from typing import List


ROOT_DIR = "/home/admin01/Projects/DexGarmentLab/Preprocess/data"
FOLDER_PREFIX = "left_1126_2"
ANNOTATION_FILENAME = "garments_box.jsonl"


def is_even_image(record: dict) -> bool:
    """Return True if the rgb filename's index is even (e.g. image_0.png -> 0)."""
    rgb = record.get("rgb")
    if not rgb:
        return False

    basename = os.path.basename(rgb)
    match = re.search(r"(\d+)", basename)
    if not match:
        return False

    idx = int(match.group(1))
    return idx % 2 == 0


def filter_file(jsonl_path: str) -> None:
    """Filter a single garments_box.jsonl file, keeping only even-index images.

    A backup file with suffix '.bak' will be created before overwriting.
    """
    if not os.path.isfile(jsonl_path):
        return

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    kept_lines: List[str] = []
    total = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        total += 1
        record = json.loads(line)
        if is_even_image(record):
            # Re-dump to ensure clean formatting
            kept_lines.append(json.dumps(record, ensure_ascii=False) + "\n")

    # Make a backup before overwriting
    backup_path = jsonl_path + ".bak"
    if not os.path.exists(backup_path):
        os.rename(jsonl_path, backup_path)
    else:
        # If backup already exists, do a safer copy-like behavior
        with open(jsonl_path, "r", encoding="utf-8") as src, open(
            backup_path, "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)

    print(
        f"Processed: {jsonl_path} | total: {total}, kept (even): {len(kept_lines)}. "
        f"Backup: {backup_path}"
    )


def main() -> None:
    for name in os.listdir(ROOT_DIR):
        folder_path = os.path.join(ROOT_DIR, name)
        if not os.path.isdir(folder_path):
            continue
        if not name.startswith(FOLDER_PREFIX):
            continue

        jsonl_path = os.path.join(folder_path, ANNOTATION_FILENAME)
        if os.path.isfile(jsonl_path):
            print(f"Filtering file: {jsonl_path}")
            filter_file(jsonl_path)


if __name__ == "__main__":
    main()


