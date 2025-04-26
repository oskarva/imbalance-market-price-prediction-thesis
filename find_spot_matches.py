#!/usr/bin/env python3
"""
find_spot_matches.py

For each area under a base directory, locate the last CV round in the specified phase directory
(default "test_rounds"), read X_train and y_train for that round, and report the percentage of
timestamps where each target variable is within a relative tolerance of the spot price. Saves
timestamp lists for each area/target to a pickle file for downstream filtering.
"""
import os
import re
import argparse
import glob
import pickle

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find percentage of target==spot (within tolerance)"
    )
    parser.add_argument(
        "--base-dir", type=str, required=True,
        help="Base directory containing area subfolders (e.g. /Volumes/T9)"
    )
    parser.add_argument(
        "--areas", type=str, default=None,
        help="Comma-separated list of areas to process (e.g. no1,no2). Defaults to all subdirs."
    )
    parser.add_argument(
        "--phase-dir", type=str, default="test_rounds",
        help="Subdirectory under each area containing the CSV rounds. Default: test_rounds"
    )
    parser.add_argument(
        "--rtol", type=float, default=0.01,
        help="Relative tolerance for spot==target comparison. Default: 0.01 (1%). Ignored if --exact is set."
    )
    parser.add_argument(
        "--atol", type=float, default=1e-6,
        help="Absolute tolerance for spot==target comparison. Default: 1e-6. Ignored if --exact is set."
    )
    parser.add_argument(
        "--exact", action="store_true",
        help="When set, use strict equality (target == spot) instead of tolerance-based comparison."
    )
    parser.add_argument(
        "--output", type=str, default="spot_target_timestamps.pkl",
        help="Path to output pickle file with timestamps per area/target."
    )
    return parser.parse_args()


def find_last_round(csv_dir, prefix):
    """Find the highest round number for files named prefix_<round>.csv in csv_dir."""
    pattern = os.path.join(csv_dir, f"{prefix}_*.csv")
    files = glob.glob(pattern)
    rounds = []
    for f in files:
        m = re.search(rf"{re.escape(prefix)}_(\d+)\.csv$", f)
        if m:
            rounds.append(int(m.group(1)))
    if not rounds:
        return None
    return max(rounds)


def main():
    args = parse_args()

    # Determine areas to process
    if args.areas:
        areas = args.areas.split(",")
    else:
        areas = [d for d in os.listdir(args.base_dir)
                 if os.path.isdir(os.path.join(args.base_dir, d))]

    results = {}

    for area in areas:
        area_dir = os.path.join(args.base_dir, area, args.phase_dir)
        if not os.path.isdir(area_dir):
            print(f"Skipping area {area}: directory not found: {area_dir}")
            continue

        # Find the last training round
        last_round = find_last_round(area_dir, "y_train")
        if last_round is None:
            print(f"  No y_train_*.csv found in {area_dir}. Skipping.")
            continue

        # Paths for X_train and y_train
        x_file = os.path.join(area_dir, f"X_train_{last_round}.csv")
        y_file = os.path.join(area_dir, f"y_train_{last_round}.csv")
        if not os.path.isfile(x_file) or not os.path.isfile(y_file):
            print(f"  Missing train files for round {last_round} in {area_dir}. Skipping.")
            continue

        # Read CSVs
        X = pd.read_csv(x_file, index_col=0, parse_dates=True)
        y = pd.read_csv(y_file, index_col=0, parse_dates=True)

        # Identify spot column in X
        spot_cols = [c for c in X.columns if "spot" in c.lower()]
        if not spot_cols:
            print(f"  No spot column found in {x_file}. Skipping.")
            continue
        spot_col = spot_cols[0]
        spot = X[spot_col]

        # Prepare storage for this area
        results[area] = {}

        # Compare each target against spot
        for target_col in y.columns:
            target = y[target_col]
            # Determine mask: strict equality or tolerance-based
            if args.exact:
                mask = (target == spot)
                desc = "exactly"
            else:
                mask = np.isclose(target, spot,
                                   rtol=args.rtol,
                                   atol=args.atol)
                desc = f"within {args.rtol*100:.1f}%"
            pct = mask.mean() * 100
            # Extract matching timestamps
            timestamps = list(target.index[mask])
            # Store
            results[area][target_col] = {
                "percentage": pct,
                "timestamps": timestamps
            }
            # Report
            print(
                f"Area {area}: target '{target_col}' matches spot {desc} at {pct:.2f}% of timestamps"
            )

    # Save results to pickle for downstream filtering
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved match timestamps to {args.output}")


if __name__ == "__main__":
    main()