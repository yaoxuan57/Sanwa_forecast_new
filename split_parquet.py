#!/usr/bin/env python3
"""
split_parquet.py

Split a Parquet dataset by rows into train/val/test and save to the same folder.

Example:
  python split_parquet.py --input /path/to/dataset.parquet --seed 42

Outputs (same directory as input):
  train.parquet (60%)
  val.parquet   (20%)
  test.parquet  (20%)
"""

import os
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to dataset.parquet")
    ap.add_argument("--train_ratio", type=float, default=0.6)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle rows before splitting")
    ap.add_argument("--prefix", default="", help="Optional prefix for output files (e.g., 'sanwa_')")
    args = ap.parse_args()

    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    in_path = args.input
    if not os.path.isfile(in_path):
        raise FileNotFoundError(in_path)

    out_dir = os.path.dirname(os.path.abspath(in_path))
    df = pd.read_parquet(in_path)

    n = len(df)
    if n < 3:
        raise ValueError(f"Dataset too small to split: {n} rows")

    idx = np.arange(n)
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(idx)

    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val  # remainder

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_path = os.path.join(out_dir, f"{args.prefix}train.parquet")
    val_path = os.path.join(out_dir, f"{args.prefix}val.parquet")
    test_path = os.path.join(out_dir, f"{args.prefix}test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print("[DONE] Split complete")
    print("Input :", in_path)
    print("Saved :")
    print(f"  {train_path}  rows={len(train_df)}")
    print(f"  {val_path}    rows={len(val_df)}")
    print(f"  {test_path}   rows={len(test_df)}")
    print("Notes:")
    print(" - Use --shuffle to shuffle rows before splitting (recommended for IID windows).")
    print(" - If you want time-ordered splits, do NOT use --shuffle.")


if __name__ == "__main__":
    main()
