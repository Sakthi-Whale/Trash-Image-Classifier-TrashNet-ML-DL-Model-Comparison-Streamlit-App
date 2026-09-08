"""
data_split.py
-------------
Splits the raw TrashNet-style dataset (one folder per class) into
train / val / test folders using a stratified split, so every class
is proportionally represented in each split.

Usage:
    python src/data_split.py --src data_raw/dataset-resized --dst data --val 0.1 --test 0.1
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def split_dataset(src: Path, dst: Path, val_frac: float, test_frac: float, seed: int = 42):
    random.seed(seed)
    classes = sorted([d.name for d in src.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in {src}")

    summary = {}
    for cls in classes:
        files = [f for f in (src / cls).iterdir() if f.suffix.lower() in IMG_EXTS]
        random.shuffle(files)

        n = len(files)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        n_train = n - n_val - n_test

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            out_dir = dst / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out_dir / f.name)

        summary[cls] = {k: len(v) for k, v in splits.items()}

    return summary, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="data_raw/dataset-resized")
    parser.add_argument("--dst", type=str, default="data")
    parser.add_argument("--val", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    summary, classes = split_dataset(src, dst, args.val, args.test, args.seed)

    print(f"Classes found ({len(classes)}): {classes}\n")
    print(f"{'class':<12}{'train':>8}{'val':>8}{'test':>8}")
    totals = {"train": 0, "val": 0, "test": 0}
    for cls, counts in summary.items():
        print(f"{cls:<12}{counts['train']:>8}{counts['val']:>8}{counts['test']:>8}")
        for k in totals:
            totals[k] += counts[k]
    print("-" * 36)
    print(f"{'TOTAL':<12}{totals['train']:>8}{totals['val']:>8}{totals['test']:>8}")


if __name__ == "__main__":
    main()
