"""
Non-homogeneous Poisson fit for Citi Bike departures, compared with the
constant-rate (homogeneous) model.

Model: for each station, one rate per (weekday|weekend, hour-of-day) cell = 48 rates.
  rate_cell = (departures in cell + ALPHA * station_avg_rate) / (days in cell + ALPHA)
  (ALPHA = 1 pseudo-day of the station's average rate; keeps empty cells from being exactly 0)

Accuracy checks, each computed for BOTH models:
  * Time-rescaling: if the model is right, rescaled gaps  Lambda(t_i) - Lambda(t_{i-1})  are iid Exp(1).
    Reported via CV of rescaled gaps (1 = exponential) and KS statistic (0 = perfect).
  * Hourly-count dispersion: mean((count - mu)^2 / mu) (1 = Poisson), in-sample.
  * Same dispersion on HELD-OUT days (fit on odd days, test on even days) to expose overfitting.

Usage: python citibike_nhpp_fit.py <csv> <outdir>
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CSV = sys.argv[1] if len(sys.argv) > 1 else "JC-202609-citibike-tripdata.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
MIN_ARRIVALS, N_SHOW, ALPHA = 30, 4, 1.0
HOM, NH = "#9a9a9a", "#3b6ea5"   # colours: homogeneous (grey), non-homogeneous (blue)

# ---------------------------------------------------------------- load data
df = pd.read_csv(CSV, usecols=["started_at", "start_station_name"])
df["started_at"] = pd.to_datetime(df["started_at"])
df = df.dropna(subset=["start_station_name"])
month = df["started_at"].dt.to_period("M").mode()[0]            # keep only the main month
n_drop = (df["started_at"].dt.to_period("M") != month).sum()
df = df[df["started_at"].dt.to_period("M") == month].sort_values("started_at")
start, n_days = month.start_time, month.days_in_month
H = n_days * 24                                                    # hourly grid length
print(f"Month {month}: {len(df)} departures ({n_drop} outside the month dropped)")

# --------------------------------------------------------- hourly grid labels
g = np.arange(H)
day, hod = g // 24, g % 24
wknd = (start + pd.to_timedelta(day, "D")).dayofweek >= 5
cell = wknd.astype(int) * 24 + hod                                 # 0..47
days_in_cell = np.array([len(set(day[(cell == c)])) for c in range(48)], dtype=float)
train = ((day + 1) % 2 == 1)                                       # odd days = training
days_in_cell_tr = np.array([len(set(day[(cell == c) & train])) for c in range(48)], dtype=float)

def cell_rates(counts, mask, nd, lam_bar):
    cc = np.bincount(cell[mask], weights=counts[mask], minlength=48)
    return (cc + ALPHA * lam_bar) / (nd + ALPHA)

def rescale(t_hours, mu_grid):
    """Integrated intensity Lambda at each arrival, then gaps (hours, piecewise-constant rate)."""
    C = np.concatenate([[0.0], np.cumsum(mu_grid)])
    gaps = np.diff(np.interp(t_hours, np.arange(H + 1), C))
    return gaps[gaps > 0]

def disp(counts, mu):
    return np.mean((counts - mu) ** 2 / mu)

# ------------------------------------------------------------ per-station fits
rows, store, long_rates = {}, {}, []
for name, d in df.groupby("start_station_name"):
    if len(d) < MIN_ARRIVALS:
        continue
    t = ((d["started_at"] - start) / pd.Timedelta(hours=1)).to_numpy()
    counts = np.bincount(np.floor(t).astype(int), minlength=H)[:H].astype(float)
    n = counts.sum(); lam = n / H

    r_all = cell_rates(counts, np.ones(H, bool), days_in_cell, lam)
    mu_nh, mu_hom = r_all[cell], np.full(H, lam)
    g_hom, g_nh = rescale(t, mu_hom), rescale(t, mu_nh)

    # held-out: fit on odd days, evaluate hourly counts on even days
    lam_tr = counts[train].sum() / train.sum()
    r_tr = cell_rates(counts, train, days_in_cell_tr, lam_tr)
    te = ~train

    rows[name] = dict(
        n_arrivals=int(n), rate_hom_per_hr=lam,
        cv_hom=g_hom.std(ddof=1) / g_hom.mean(), cv_nhpp=g_nh.std(ddof=1) / g_nh.mean(),
        ks_hom=stats.kstest(g_hom, "expon").statistic, ks_nhpp=stats.kstest(g_nh, "expon").statistic,
        disp_hom=disp(counts, mu_hom), disp_nhpp=disp(counts, mu_nh),
        disp_ho_hom=disp(counts[te], lam_tr), disp_ho_nhpp=disp(counts[te], r_tr[cell[te]]))
    store[name] = dict(g_hom=g_hom, g_nh=g_nh, counts=counts, mu_nh=mu_nh, mu_hom=mu_hom, r=r_all)
    for c in range(48):
        long_rates.append((name, "weekend" if c >= 24 else "weekday", c % 24, r_all[c]))

res = pd.DataFrame(rows).T.sort_values("n_arrivals", ascending=False)
res.index.name = "station"
res.to_csv(f"{OUT}/station_fit_comparison.csv")
pd.DataFrame(long_rates, columns=["station", "daytype", "hour", "rate_per_hour"]).to_csv(
    f"{OUT}/station_rates_by_hour.csv", index=False)

print(f"\nStations fitted: {len(res)}   (medians across stations; target value in brackets)")
for lab, a, b, tgt in [("CV of rescaled gaps", "cv_hom", "cv_nhpp", 1), ("KS statistic", "ks_hom", "ks_nhpp", 0),
                       ("Count dispersion (in-sample)", "disp_hom", "disp_nhpp", 1),
                       ("Count dispersion (held-out days)", "disp_ho_hom", "disp_ho_nhpp", 1)]:
    print(f"  {lab:34s} constant {res[a].astype(float).median():6.2f} -> hourly {res[b].astype(float).median():6.2f}   [{tgt}]")

# -------------------------------------------- Fig A: detailed fits, busiest stations
show = res.index[:N_SHOW]
fig, axes = plt.subplots(3, N_SHOW, figsize=(4.2 * N_SHOW, 11))
for j, name in enumerate(show):
    s, r = store[name], res.loc[name]
    ax = axes[0, j]                                              # ECDF of rescaled gaps vs Exp(1)
    for gaps, col, lab in [(s["g_hom"], HOM, f"constant rate (CV {r.cv_hom:.2f})"),
                           (s["g_nh"], NH, f"hour-of-day rate (CV {r.cv_nhpp:.2f})")]:
        x = np.sort(gaps); ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col, label=lab)
    xx = np.linspace(0, 5, 200); ax.plot(xx, 1 - np.exp(-xx), "r--", label="Exp(1)")
    ax.set(xlim=(0, 5), title=f"{name}\n(n={int(r.n_arrivals)})", xlabel="rescaled inter-arrival (mean = 1)", ylabel="CDF")
    ax.legend(fontsize=7)

    ax = axes[1, j]                                              # Q-Q log-log
    for gaps, col in [(s["g_hom"], HOM), (s["g_nh"], NH)]:
        x = np.sort(gaps); q = stats.expon.ppf((np.arange(1, len(x) + 1) - .5) / len(x))
        ax.plot(q, x, ".", ms=3, color=col)
    ax.plot([1e-3, 10], [1e-3, 10], "r--"); ax.set(xscale="log", yscale="log", xlabel="Exp(1) quantile",
                                                    ylabel="observed quantile", title="Q-Q (log-log)")

    ax = axes[2, j]                                              # hourly count distribution
    c = s["counts"]; k = np.arange(0, int(c.max()) + 1)
    ax.hist(c, bins=np.arange(-0.5, c.max() + 1.5, max(1, int(c.max()) // 25)), density=True,
            color="#c9d6e8", label="observed")
    ax.plot(k, stats.poisson.pmf(k, s["mu_hom"][0]), color=HOM, marker=".", label=f"constant (disp {r.disp_hom:.1f})")
    ax.plot(k, np.mean([stats.poisson.pmf(k, m) for m in s["mu_nh"]], axis=0), color=NH, marker=".",
            label=f"hour-of-day (disp {r.disp_nhpp:.1f})")
    ax.set(xlabel="departures per hour", ylabel="probability", title="Hourly counts vs model"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_nhpp_station_fits.png", dpi=150); plt.close()

# ----------------------------------- Fig B: before/after across all stations
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, (a, b, title, tgt, xl) in zip(axes, [
        ("cv_hom", "cv_nhpp", "Rescaled-gap CV", 1, "CV (1 = exponential)"),
        ("ks_hom", "ks_nhpp", "KS statistic on rescaled gaps", 0, "KS statistic (0 = perfect)"),
        ("disp_hom", "disp_nhpp", "Hourly-count dispersion, in-sample", 1, "var/mean (1 = Poisson)"),
        ("disp_ho_hom", "disp_ho_nhpp", "Hourly-count dispersion, held-out days", 1, "var/mean (1 = Poisson)")]):
    va, vb = res[a].astype(float), res[b].astype(float)
    bins = np.linspace(min(va.min(), vb.min(), tgt), np.percentile(np.r_[va, vb], 99) * 1.05, 30)
    ax.hist(va, bins, color=HOM, alpha=.7, label=f"constant rate (median {va.median():.2f})")
    ax.hist(vb, bins, color=NH, alpha=.7, label=f"hour-of-day rate (median {vb.median():.2f})")
    ax.axvline(tgt, color="r", ls="--"); ax.set(title=title, xlabel=xl, ylabel="# stations"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_nhpp_goodness_summary.png", dpi=150); plt.close()

# ------------------------------------------------- Fig C: fitted rate profiles
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
tot = np.sum([v["r"] for v in store.values()], axis=0)          # system departures/hr by cell
ax[0].plot(range(24), tot[:24], color=NH, marker="o", label="weekday")
ax[0].plot(range(24), tot[24:], color="#d97b29", marker="o", label="weekend")
ax[0].set(xlabel="hour of day", ylabel="departures / hour (all stations)", title="System-wide fitted rate"); ax[0].legend()
top = res.index[:15]
M = np.array([store[n]["r"][:24] for n in top])
im = ax[1].imshow(M, aspect="auto", cmap="Blues")
ax[1].set(yticks=range(15), yticklabels=top, xlabel="hour of day", title="Weekday rate λ(hour), 15 busiest stations")
ax[1].tick_params(axis="y", labelsize=8); plt.colorbar(im, ax=ax[1], label="departures / hour")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_nhpp_rate_profiles.png", dpi=150); plt.close()
