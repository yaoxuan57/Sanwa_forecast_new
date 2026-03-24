#!/usr/bin/env python3
"""
make_windows_to_parquet.py

Build sliding windows for 1-step-ahead forecasting.

Input  : (N, C) time-ordered table
Output : Parquet with columns:
         - samples: (C, window) per row (list of lists)
         - labels : (C, 1) per row (list of lists)

Optional grouping so windows don't cross boundaries.
"""

import os
import argparse
from typing import List, Optional
import numpy as np
import pandas as pd


def parse_list(s: Optional[str]) -> List[str]:
    if s is None or s.strip() == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {ext} (use .csv or .parquet)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input .csv or .parquet")
    ap.add_argument("--columns", required=True, help="Comma-separated feature columns")
    ap.add_argument("--target_columns", default="", help="Optional comma-separated target columns (default = columns)")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--time_col", default="", help="Optional time column to sort by")
    ap.add_argument("--groupby", default="", help="Optional comma-separated group columns")
    ap.add_argument("--dropna", action="store_true")
    ap.add_argument("--out_dir", default="windows_out")
    ap.add_argument("--out_parquet", default="dataset.parquet", help="Output parquet filename")
    args = ap.parse_args()

    feat_cols = parse_list(args.columns)
    tgt_cols = parse_list(args.target_columns) if args.target_columns.strip() else feat_cols
    group_cols = parse_list(args.groupby)
    time_col = args.time_col.strip() if args.time_col else ""

    df = load_table(args.input)

    keep_cols = list(dict.fromkeys(group_cols + ([time_col] if time_col else []) + feat_cols + tgt_cols))
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in file: {missing}")

    df = df[keep_cols].copy()

    # sort
    if time_col:
        df = df.sort_values(group_cols + [time_col] if group_cols else [time_col]).reset_index(drop=True)

    # numeric conversion
    for c in set(feat_cols + tgt_cols):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if args.dropna:
        df = df.dropna(subset=list(set(feat_cols + tgt_cols))).reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)

    samples_rows = []
    labels_rows = []

    def process_block(block: pd.DataFrame):
        feat_arr = block[feat_cols].to_numpy(dtype=np.float32)  # (N, C_feat)
        tgt_arr  = block[tgt_cols].to_numpy(dtype=np.float32)   # (N, C_tgt)

        N = len(block)
        max_start = N - args.window - args.horizon + 1
        if max_start <= 0:
            return

        for start in range(max_start):
            x_win = feat_arr[start : start + args.window]  # (window, C_feat)
            y_t   = tgt_arr[start + args.window + args.horizon - 1]  # (C_tgt,)

            # store as python lists for parquet
            # samples: (C, window)
            samples_rows.append(x_win.T.tolist())
            # labels : (C, 1)
            labels_rows.append(y_t.reshape(-1, 1).tolist())

    if group_cols:
        for _, g in df.groupby(group_cols, sort=False):
            process_block(g)
    else:
        process_block(df)

    if len(samples_rows) == 0:
        raise ValueError("No windows were created. Check window/horizon sizes and grouping.")

    out_df = pd.DataFrame({
        "samples": samples_rows,
        "labels": labels_rows,
    })

    out_path = os.path.join(args.out_dir, args.out_parquet)
    out_df.to_parquet(out_path, index=False)

    print("[DONE] Saved parquet:")
    print(" ", out_path)
    print(" rows:", len(out_df))
    print(" sample shape per row: (C, window) =", (len(samples_rows[0]), len(samples_rows[0][0])))
    print(" label  shape per row: (C, 1)      =", (len(labels_rows[0]), len(labels_rows[0][0])))


if __name__ == "__main__":
    main()

# python forecast_pre_process.py --input "/scratch/prj0000000262/Sanwa_forecast/Sanwa_forecast_ft/injection_molding_machine_12.17_YX.csv" --columns "max_injection_pressure,switchover_pressure,end_of_packing_stroke,plastification_time"