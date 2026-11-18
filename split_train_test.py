"""
80/20 train/test split of the Citi Bike trip data, by whole days.

Every calendar day is assigned entirely to train or to test (by the date of
`started_at`), so no day contributes rides to both sets. The split is done
separately for each month, and within each month separately for weekdays and
weekends: 20% of the month's weekdays and 20% of its weekend days go to test.
That keeps the weekday/weekend mix of train and test the same as the month's.
The per-month pieces are stacked into one combined train file and one combined
test file.

Months come from the folder names in data/ (e.g. data/202608-citibike-tripdata/).
Files whose names start with "JC" (Jersey City) are ignored. Rides that start
outside the folder months (e.g. July 31 rides that ended up in the August
files) are dropped, because those days are only partially covered.

Files are streamed in chunks, so the full ~3 GB of data never has to fit in
memory. Values are read and written as raw strings, so output rows are
identical to the input rows plus `source_month` (e.g. "202608") and
`day_type` ("weekday"/"weekend") columns. A list of which days went where is
written to <out-dir>/day_assignment.csv.

Usage:
    python split_train_test.py [--data-dir data] [--out-dir data/splits]
                               [--test-frac 0.2] [--seed 451]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 1_000_000
MONTH_RE = re.compile(r"^(\d{6})-citibike-tripdata")


def month_files(data_dir: Path) -> dict[str, list[Path]]:
    """Map 'YYYYMM' -> sorted list of CSVs for that month (JC files excluded)."""
    months: dict[str, list[Path]] = {}
    for csv in sorted(data_dir.rglob("*.csv")):
        if csv.name.startswith("JC") or data_dir / "splits" in csv.parents:
            continue
        m = MONTH_RE.match(csv.name)
        if m:
            months.setdefault(m.group(1), []).append(csv)
    return months


def assign_days(months, test_frac: float, rng) -> pd.DataFrame:
    """Pick test days per month, stratified by weekday/weekend."""
    rows = []
    for month in sorted(months):
        days = pd.period_range(pd.Period(month, "M").start_time,
                               pd.Period(month, "M").end_time, freq="D")
        df = pd.DataFrame({"date": days.strftime("%Y-%m-%d"),
                           "source_month": month,
                           "day_type": np.where(days.dayofweek >= 5, "weekend", "weekday"),
                           "split": "train"})
        for _, idx in df.groupby("day_type").groups.items():
            n_test = int(round(len(idx) * test_frac))
            df.loc[rng.choice(idx, size=n_test, replace=False), "split"] = "test"
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--out-dir", default="data/splits", type=Path)
    ap.add_argument("--test-frac", default=0.2, type=float)
    ap.add_argument("--seed", default=451, type=int)
    args = ap.parse_args()

    months = month_files(args.data_dir)
    if not months:
        raise SystemExit(f"No monthly Citi Bike CSVs found under {args.data_dir}")

    days = assign_days(months, args.test_frac, np.random.default_rng(args.seed))
    lookup = days.set_index("date")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": args.out_dir / "train.csv", "test": args.out_dir / "test.csv"}
    for p in paths.values():
        p.unlink(missing_ok=True)

    header_written = False
    ride_counts = {}  # date -> number of rides
    n_dropped = 0
    for month, files in sorted(months.items()):
        for f in files:
            for chunk in pd.read_csv(f, dtype=str, keep_default_na=False,
                                     chunksize=CHUNK):
                date = chunk["started_at"].str[:10]
                keep = date.isin(lookup.index)
                n_dropped += int((~keep).sum())
                chunk, date = chunk[keep], date[keep]
                info = lookup.loc[date]
                chunk = chunk.assign(source_month=info["source_month"].to_numpy(),
                                     day_type=info["day_type"].to_numpy())
                split = info["split"].to_numpy()
                for name, path in paths.items():
                    chunk[split == name].to_csv(path, mode="a", index=False,
                                                header=not header_written)
                header_written = True
                for d, c in date.value_counts().items():
                    ride_counts[d] = ride_counts.get(d, 0) + c

    days["rides"] = days["date"].map(ride_counts).fillna(0).astype(int)
    days.to_csv(args.out_dir / "day_assignment.csv", index=False)

    # ---- summary
    print(f"{'month':<7} {'type':<8} {'train days':>10} {'test days':>9}"
          f" {'test day %':>10} {'test ride %':>11}")
    for (month, dtype), g in days.groupby(["source_month", "day_type"]):
        t = g[g.split == "test"]
        print(f"{month:<7} {dtype:<8} {len(g) - len(t):>10} {len(t):>9}"
              f" {len(t) / len(g):>10.1%} {t.rides.sum() / g.rides.sum():>11.1%}")
    for month, g in days.groupby("source_month"):
        t = g[g.split == "test"]
        print(f"{month} total: {len(g)} days ({len(t)} test, {len(t) / len(g):.1%}); "
              f"{g.rides.sum():,} rides ({t.rides.sum():,} test, "
              f"{t.rides.sum() / g.rides.sum():.1%}); "
              f"weekend share train {(g[g.split == 'train'].day_type == 'weekend').mean():.1%}"
              f" / test {(t.day_type == 'weekend').mean():.1%}")
    tr, te = days[days.split == "train"].rides.sum(), days[days.split == "test"].rides.sum()
    print(f"TOTAL: train {tr:,} rides, test {te:,} rides ({te / (tr + te):.1%} test); "
          f"dropped {n_dropped:,} rides starting outside the folder months")
    print(f"Wrote {paths['train']}, {paths['test']}, {args.out_dir / 'day_assignment.csv'}")


if __name__ == "__main__":
    main()
