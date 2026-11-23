"""
Our Model
Events:
Arrival of customer to take bike at station i at time tA and return to station j
Arrival of customer to return bike at station i at time tB
Rebalancing of Station i at time tR according to the prescribed policy
State Space:
number of bikes bi at each station
Counters:
Total time that station i is empty for
Number of customers that arrive to take a bike at an empty station i
Number of customers that arrive to return a bike at a full station i

Modeling assumptions:
split each arc into "leaves station i" × "goes to j"
model interarrivals at each station according a gamma distribution

Implementation (v1: no rebalancing; time unit = hours)
------------------------------------------------------
Pickup arrivals at station i form a gamma renewal process in "operational time":
  * Each gap E ~ Gamma(shape k_i, scale 1/k_i), so its mean is 1. It is laid out
    against the station's hourly rate lambda_i(h) (48 rates: weekday/weekend x hour):
    the next arrival is at t' where the integral of lambda_i from t to t' equals E.
    This is how the gaps were fitted (time rescaling), it gives the right number of
    arrivals per hour on average, and gaps can cross hour boundaries cleanly.
  * lambda_i(h) comes from output/station_rates_by_hour.csv and k_i from
    output/station_distribution_fits.csv (gamma_shape), both produced by
    citibike_nhpp_fit.py from the train days. k_i < 1 means burstier than Poisson.
Each pickup picks a destination j with probability p_ij(block), where the blocks are
  night 0-6, AM peak 6-10, midday 10-16, PM peak 16-20, evening 20-24, x weekday/weekend:
  p_ij(b) = (n_ij(b) + alpha * p_ij(all hours)) / (n_i(b) + alpha), alpha chosen by
  test-day log-likelihood. Only destinations seen from i in the train days are possible.
Bike type: e-bike with probability = station i's e-bike share in the train days.
Trip time: 1.4 x straight-line distance between the stations / average speed of that
  bike type (total distance / total time over train trips of 1-180 min). Round trips
  (j = i) take the median round-trip duration instead.
Station locations and dock capacities come from the Citi Bike GBFS station_information
  feed (saved to data/station_information.json), matched on station id, then on name.
  Unmatched stations use their median trip coordinates; missing/zero capacity uses the
  median capacity.
Customers who find station i empty leave (lost pickup). Riders who find station j full
  ride to the nearest of its 10 nearest stations that has a free dock.
Trips dropped when estimating: no end station, or start/end not a modeled station.

Usage: python main.py [--days 1] [--start 2026-09-14] [--fill 0.5] [--seed 451] [--rebuild]
"""
import argparse
import heapq
import itertools
import json
import math
import pickle
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = Path("data/splits")
OUTDIR = Path("output")
RATES_CSV = OUTDIR / "station_rates_by_hour.csv"
FITS_CSV = OUTDIR / "station_distribution_fits.csv"
STATION_INFO = Path("data/station_information.json")
STATION_INFO_URL = "https://gbfs.lyft.com/gbfs/1.1/bkn/en/station_information.json"
CACHE = SPLITS / "des_params.pkl"          # derived from the splits, so it lives (ignored) beside them

DETOUR = 1.4                                # street distance = DETOUR x straight line
BLOCK_OF_HOUR = np.array([0] * 6 + [1] * 4 + [2] * 6 + [3] * 4 + [4] * 4)  # 5 blocks per day type
N_BLOCKS = 10                               # 5 blocks x weekday/weekend
ALPHAS = [10, 30, 100, 300, 1000, 3000, 10000, 30000]    # candidate smoothing strengths
MIN_DUR, MAX_DUR = 1, 180                   # minutes; trips used to estimate speed
K_NEAREST = 10                              # stations tried when a return finds a full station
EARTH_KM = 6371.0


# ====================================================================== estimation
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(h))


def block_index(ts):
    """Destination-choice block (0..9) for a Series of timestamps."""
    return (ts.dt.dayofweek >= 5).to_numpy().astype(int) * 5 + BLOCK_OF_HOUR[ts.dt.hour.to_numpy()]


def station_locations(stations, train):
    """Coordinates + capacity per modeled station from the GBFS feed (fallbacks noted in docstring)."""
    if not STATION_INFO.exists():
        print(f"Downloading station information -> {STATION_INFO}")
        urllib.request.urlretrieve(STATION_INFO_URL, STATION_INFO)
    info = pd.DataFrame(json.load(open(STATION_INFO))["data"]["stations"])
    by_id = info.drop_duplicates("short_name").set_index("short_name")
    by_name = info.drop_duplicates("name").set_index("name")

    ids = train.groupby("start_station_name", observed=True)["start_station_id"].agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else None)
    trip_xy = train.groupby("start_station_name", observed=True)[["start_lat", "start_lng"]].median()

    lat, lon, cap, how = [], [], [], {"id": 0, "name": 0, "trip coords": 0}
    for s in stations:
        sid = ids.get(s)
        if sid in by_id.index:
            row, src = by_id.loc[sid], "id"
        elif s in by_name.index:
            row, src = by_name.loc[s], "name"
        else:
            row, src = None, "trip coords"
        how[src] += 1
        if row is None:
            lat.append(trip_xy.loc[s, "start_lat"]); lon.append(trip_xy.loc[s, "start_lng"]); cap.append(0)
        else:
            lat.append(row["lat"]); lon.append(row["lon"]); cap.append(int(row["capacity"]))
    cap = np.array(cap)
    med = int(np.median(cap[cap > 0]))
    n_fix = int((cap <= 0).sum())
    cap[cap <= 0] = med
    print(f"  station locations matched by {how}; {n_fix} missing/zero capacities set to median {med}")
    return np.array(lat), np.array(lon), cap


def build_params():
    t0 = time.time()
    for f in (RATES_CSV, FITS_CSV):
        if not f.exists():
            raise SystemExit(f"{f} not found - run citibike_nhpp_fit.py first")

    # ---- arrival process: hourly rates (48 cells) + gamma shape per station
    rates = pd.read_csv(RATES_CSV)
    rates["cell"] = (rates.daytype == "weekend") * 24 + rates.hour
    R = rates.pivot(index="station", columns="cell", values="rate_per_hour").sort_index()
    stations = R.index.to_list()
    idx = {s: i for i, s in enumerate(stations)}
    shape = pd.read_csv(FITS_CSV, index_col="station")["gamma_shape"].reindex(stations)
    shape = shape.fillna(shape.median()).clip(0.2, 5).to_numpy()

    # ---- train trips
    cols = ["rideable_type", "started_at", "ended_at", "start_station_name", "start_station_id",
            "end_station_name", "start_lat", "start_lng"]
    tr = pd.read_csv(SPLITS / "train.csv", usecols=cols, dtype={"start_station_id": str})
    lat, lon, cap = station_locations(stations, tr)

    n0 = len(tr)
    tr = tr[tr.start_station_name.isin(idx) & tr.end_station_name.isin(idx)]
    print(f"  train trips kept {len(tr):,} of {n0:,} (dropped: no end station or not a modeled station)")
    o = tr.start_station_name.map(idx).to_numpy()
    d = tr.end_station_name.map(idx).to_numpy()
    start = pd.to_datetime(tr.started_at, format="ISO8601")
    dur_min = ((pd.to_datetime(tr.ended_at, format="ISO8601") - start).dt.total_seconds() / 60).to_numpy()
    ebike = (tr.rideable_type == "electric_bike").to_numpy()
    blk = block_index(start)

    # ---- bike type share per origin, speeds, round-trip duration
    ebike_share = np.bincount(o, weights=ebike, minlength=len(stations)) / np.bincount(o, minlength=len(stations))
    km = DETOUR * haversine_km(lat[o], lon[o], lat[d], lon[d])
    ok = (o != d) & (dur_min >= MIN_DUR) & (dur_min <= MAX_DUR)
    speed = {b: km[ok & (ebike == b)].sum() / (dur_min[ok & (ebike == b)].sum() / 60) for b in (False, True)}
    round_trip_hr = np.median(dur_min[o == d]) / 60
    print(f"  speeds: classic {speed[False]:.1f} km/h, e-bike {speed[True]:.1f} km/h; "
          f"round trip {round_trip_hr * 60:.1f} min; e-bike share {ebike.mean():.1%}")

    # ---- destination choice by block, smoothed toward the all-hours shares
    g = pd.DataFrame({"o": o, "d": d, "b": blk})
    nijb = g.groupby(["o", "d", "b"]).size()
    nij, ni, nib = g.groupby(["o", "d"]).size(), g.groupby("o").size(), g.groupby(["o", "b"]).size()

    te = pd.read_csv(SPLITS / "test.csv", usecols=["started_at", "start_station_name", "end_station_name"])
    te = te[te.start_station_name.isin(idx) & te.end_station_name.isin(idx)]
    te = pd.DataFrame({"o": te.start_station_name.map(idx).to_numpy(),
                       "d": te.end_station_name.map(idx).to_numpy(),
                       "b": block_index(pd.to_datetime(te.started_at, format="ISO8601"))})
    n_te = len(te)
    te = te.merge(nij.rename("nij").reset_index(), on=["o", "d"])            # seen arcs only
    te = te.merge(nijb.rename("nijb").reset_index(), on=["o", "d", "b"], how="left").fillna({"nijb": 0})
    te["ni"] = ni.reindex(te.o).to_numpy()
    te["nib"] = nib.reindex(pd.MultiIndex.from_arrays([te.o, te.b])).fillna(0).to_numpy()
    print(f"  test trips on arcs never seen in train: {1 - len(te) / n_te:.1%} (zero probability in the model)")
    ll = {a: np.log((te.nijb + a * te.nij / te.ni) / (te.nib + a)).mean() for a in ALPHAS}
    alpha = max(ll, key=ll.get)
    print("  destination log-lik per test trip: all-hours shares "
          f"{np.log(te.nij / te.ni).mean():.4f}; by block, alpha = "
          + ", ".join(f"{a}: {v:.4f}" for a, v in ll.items()) + f"  -> alpha = {alpha}")

    wide = nijb.unstack("b", fill_value=0).reindex(columns=range(N_BLOCKS), fill_value=0)
    dest, cum = [None] * len(stations), [None] * len(stations)
    for i, sub in wide.groupby(level="o"):
        cnt = sub.to_numpy().T.astype(float)                               # (blocks, destinations)
        phat = cnt.sum(0) / cnt.sum()
        p = (cnt + alpha * phat) / (cnt.sum(1, keepdims=True) + alpha)
        c = np.cumsum(p, axis=1); c[:, -1] = 1.0
        dest[i], cum[i] = sub.index.get_level_values("d").to_numpy(), c
    missing = [stations[i] for i in range(len(stations)) if dest[i] is None]
    for i in (idx[s] for s in missing):                                    # never happens in practice
        dest[i], cum[i] = np.array([i]), np.ones((N_BLOCKS, 1))

    # ---- nearest stations (for returns to full stations)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(D, np.inf)
    nearest = np.argsort(D, axis=1)[:, :K_NEAREST]

    P = dict(stations=stations, rates=R.to_numpy(), gamma_shape=shape, lat=lat, lon=lon,
             capacity=cap, ebike_share=ebike_share, speed_classic=speed[False],
             speed_ebike=speed[True], round_trip_hr=round_trip_hr, alpha=alpha,
             dest=dest, cum=cum, nearest=nearest)
    with open(CACHE, "wb") as f:
        pickle.dump(P, f)
    print(f"  parameters built in {time.time() - t0:.0f}s, cached to {CACHE}")
    return P


def load_params(rebuild=False):
    if CACHE.exists() and not rebuild:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    print("Building model parameters from the train days ...")
    return build_params()


# ====================================================================== simulation
PICKUP, RETURN = 0, 1


def simulate(P, days, start, fill, seed):
    rng = np.random.default_rng(seed)
    n, H = len(P["stations"]), days * 24
    dates = pd.date_range(start, periods=days, freq="D")
    weekend = np.repeat((dates.dayofweek >= 5).astype(int), 24)
    hod = np.tile(np.arange(24), days)
    R = P["rates"][:, weekend * 24 + hod]                  # (stations, sim hours) pickup rates
    block_of = weekend * 5 + BLOCK_OF_HOUR[hod]
    shape, cap = P["gamma_shape"], P["capacity"]
    dest, cum, near = P["dest"], P["cum"], P["nearest"]
    lat, lon = np.radians(P["lat"]), np.radians(P["lon"])
    coslat = np.cos(lat)
    v = {False: P["speed_classic"] / DETOUR, True: P["speed_ebike"] / DETOUR}  # straight-line km/h

    def travel_hr(i, j, ebike):
        if i == j:
            return P["round_trip_hr"]
        h = (math.sin((lat[j] - lat[i]) / 2) ** 2
             + coslat[i] * coslat[j] * math.sin((lon[j] - lon[i]) / 2) ** 2)
        return 2 * EARTH_KM * math.asin(math.sqrt(h)) / v[ebike]

    def next_pickup(i, t):
        """Lay a mean-1 gamma gap against station i's hourly rates, starting at t."""
        E = rng.gamma(shape[i], 1 / shape[i])
        r, h = R[i], int(t)
        while h < H:
            room = (h + 1 - t) * r[h]
            if E <= room:
                return t + E / r[h]
            E -= room
            h += 1
            t = float(h)
        return math.inf

    # state and counters
    bikes = np.round(fill * cap).astype(int)
    bikes0 = bikes.copy()
    empty_since = np.where(bikes == 0, 0.0, np.nan)
    empty_hours = np.zeros(n)
    lost_pickups = np.zeros(n, int)
    full_returns = np.zeros(n, int)
    pickups = np.zeros(n, int)
    returns = np.zeros(n, int)

    heap, seq = [], itertools.count()
    for i in range(n):
        t1 = next_pickup(i, 0.0)
        if t1 < H:
            heapq.heappush(heap, (t1, next(seq), PICKUP, i, False))

    n_events = 0
    while heap:
        t, _, kind, i, ebike = heapq.heappop(heap)
        if t >= H:
            break
        n_events += 1
        if kind == PICKUP:
            t1 = next_pickup(i, t)
            if t1 < H:
                heapq.heappush(heap, (t1, next(seq), PICKUP, i, False))
            if bikes[i] == 0:                               # customer finds station empty: lost
                lost_pickups[i] += 1
                continue
            bikes[i] -= 1
            pickups[i] += 1
            if bikes[i] == 0:
                empty_since[i] = t
            j = dest[i][np.searchsorted(cum[i][block_of[int(t)]], rng.random(), side="right")]
            eb = rng.random() < P["ebike_share"][i]
            heapq.heappush(heap, (t + travel_hr(i, j, eb), next(seq), RETURN, j, eb))
        else:
            if bikes[i] >= cap[i]:                          # full: ride on to nearest open dock
                full_returns[i] += 1
                k = next((k for k in near[i] if bikes[k] < cap[k]), near[i][0])
                heapq.heappush(heap, (t + travel_hr(i, k, ebike), next(seq), RETURN, k, ebike))
                continue
            if bikes[i] == 0:
                empty_hours[i] += t - empty_since[i]
                empty_since[i] = np.nan
            bikes[i] += 1
            returns[i] += 1

    still_empty = bikes == 0
    empty_hours[still_empty] += H - empty_since[still_empty]
    in_transit = int(bikes0.sum() - bikes.sum())
    res = pd.DataFrame(dict(station=P["stations"], capacity=cap, bikes_start=bikes0, bikes_end=bikes,
                            pickups=pickups, lost_pickups=lost_pickups, returns=returns,
                            full_returns=full_returns, hours_empty=empty_hours))
    return res, n_events, in_transit


def main():
    ap = argparse.ArgumentParser(description="Citi Bike discrete event simulation (no rebalancing)")
    ap.add_argument("--days", type=int, default=1, help="simulation length in days")
    ap.add_argument("--start", default="2026-09-14", help="first simulated date (sets weekday/weekend)")
    ap.add_argument("--fill", type=float, default=0.5, help="initial bikes as a fraction of capacity")
    ap.add_argument("--seed", type=int, default=451)
    ap.add_argument("--rebuild", action="store_true", help="re-estimate parameters from the train days")
    args = ap.parse_args()

    P = load_params(args.rebuild)
    t0 = time.time()
    res, n_events, in_transit = simulate(P, args.days, args.start, args.fill, args.seed)
    wall = time.time() - t0

    H = args.days * 24
    attempts = res.pickups.sum() + res.lost_pickups.sum()
    print(f"\nSimulated {args.days} day(s) from {args.start}: {len(res):,} stations, "
          f"{res.capacity.sum():,} docks, {res.bikes_start.sum():,} bikes "
          f"({args.fill:.0%} full); {n_events:,} events in {wall:.1f}s")
    print(f"  pickup attempts      {attempts:>10,}")
    print(f"  trips started        {res.pickups.sum():>10,}")
    print(f"  lost (empty station) {res.lost_pickups.sum():>10,}  ({res.lost_pickups.sum() / attempts:.1%} of attempts)")
    print(f"  returns at full stn  {res.full_returns.sum():>10,}  (rode on to a nearby station)")
    print(f"  bikes in transit at end {in_transit:>7,}")
    print(f"  station-hours empty  {res.hours_empty.sum():>10,.0f}  "
          f"({res.hours_empty.sum() / (len(res) * H):.1%} of all station-hours); "
          f"stations empty at end: {(res.bikes_end == 0).sum():,}, full at end: "
          f"{(res.bikes_end >= res.capacity).sum():,}")
    print("\n  Most lost pickups:")
    print(res.nlargest(10, "lost_pickups")[["station", "capacity", "bikes_start", "lost_pickups",
                                            "hours_empty"]].to_string(index=False))
    out = OUTDIR / "des_station_results.csv"
    res.sort_values("lost_pickups", ascending=False).to_csv(out, index=False)
    print(f"\nPer-station results written to {out}")


if __name__ == "__main__":
    main()
