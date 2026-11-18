"""
Non-homogeneous Poisson fit for Citi Bike departures, compared with the
constant-rate (homogeneous) model, plus a comparison of candidate distributions.

Data: the day-based train/test split made by split_train_test.py (data/splits/):
  train.csv, test.csv and day_assignment.csv (which whole days are train vs test,
  stratified by weekday/weekend within each month). Every model is FITTED ON TRAIN
  DAYS ONLY and the held-out numbers are computed on the TEST DAYS.

Model: for each station, one rate per (weekday|weekend, hour-of-day) cell = 48 rates.
  rate_cell = (train departures in cell + ALPHA * station_avg_rate) / (train days in cell + ALPHA)
  (ALPHA = 1 pseudo-day of the station's average rate; keeps empty cells from being exactly 0)

Accuracy checks, each computed for BOTH models:
  * Time-rescaling: if the model is right, rescaled gaps  Lambda(t_i) - Lambda(t_{i-1})  are iid Exp(1).
    Reported via CV of rescaled gaps (1 = exponential) and KS statistic (0 = perfect), on train days.
  * Hourly-count dispersion: mean((count - mu)^2 / mu) (1 = Poisson), on train days (in-sample).
  * Same dispersion on the held-out TEST days, to expose overfitting.

Which distribution fits best (per station; written to station_distribution_fits.csv):
  * Inter-arrival times, after rescaling by the hour-of-day rate (Exp(1) if the NHPP is right):
    exponential, gamma, Weibull, lognormal (all with location 0).
  * Hourly departure counts: Poisson vs negative binomial (NB has one extra size
    parameter r, var = mu + mu^2/r), each with a constant or an hour-of-day mean.
  Each candidate is fitted on the train days and scored two ways:
    - AIC = 2*(# params) - 2*log-likelihood on the train days (lower = better);
    - held-out log-likelihood: average log-likelihood on the test days (higher = better).
      This is the main criterion, since it can't reward overfitting.
  Gaps are rescaled with zero intensity on the other set's days, so a gap that skips
  over a test day (for train) or a train day (for test) only counts its own set's hours.

Usage: python citibike_nhpp_fit.py [splits_dir] [outdir]      (defaults: data/splits  output)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize, stats

SPLITS = sys.argv[1] if len(sys.argv) > 1 else "data/splits"
OUT = sys.argv[2] if len(sys.argv) > 2 else "output"
os.makedirs(OUT, exist_ok=True)
MIN_ARRIVALS, MIN_TEST, N_SHOW, ALPHA = 30, 10, 4, 1.0   # min train / test departures per station
HOM, NH = "#9a9a9a", "#3b6ea5"   # colours: homogeneous (grey), non-homogeneous (blue)

# ---------------------------------------------------------------- load data
T0 = time.time()
days = pd.read_csv(f"{SPLITS}/day_assignment.csv", parse_dates=["date"]).sort_values("date")
start = days["date"].iloc[0]
n_days = (days["date"].iloc[-1] - start).days + 1
H = n_days * 24                                                    # hourly grid length

def load(split):
    d = pd.read_csv(f"{SPLITS}/{split}.csv", usecols=["started_at", "start_station_name"],
                    dtype={"start_station_name": "category"})
    d = d.dropna(subset=["start_station_name"])
    t = (pd.to_datetime(d["started_at"], format="ISO8601") - start) / pd.Timedelta(hours=1)
    return pd.DataFrame({"station": d["start_station_name"].astype(str), "t": t.to_numpy()})

df = pd.concat([load("train"), load("test")], ignore_index=True)
df = df[(df["t"] >= 0) & (df["t"] < H)]
df["station"] = df["station"].astype("category")
print(f"{start:%Y-%m-%d} .. {days['date'].iloc[-1]:%Y-%m-%d}: {n_days} days "
      f"({(days.split == 'train').sum()} train / {(days.split == 'test').sum()} test), "
      f"{len(df):,} departures, {df.station.nunique():,} stations   [loaded in {time.time() - T0:.0f}s]")

# --------------------------------------------------------- hourly grid labels
g = np.arange(H)
day, hod = g // 24, g % 24
wknd = (start + pd.to_timedelta(day, "D")).dayofweek >= 5
cell = wknd.astype(int) * 24 + hod                                 # 0..47
split_of_day = days.set_index("date")["split"].reindex(
    pd.date_range(start, periods=n_days, freq="D")).to_numpy()
train = split_of_day[day] == "train"                               # hour-level masks
te = split_of_day[day] == "test"
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

# ------------------------------------------------- candidate distributions
# Inter-arrival (rescaled gap) candidates: scipy distribution, # free params (loc fixed at 0)
GAP_DISTS = {"exponential": (stats.expon, 1), "gamma": (stats.gamma, 2),
             "weibull": (stats.weibull_min, 2), "lognormal": (stats.lognorm, 2)}
# Hourly-count candidates: (count family, mean model)
COUNT_MODELS = {"poisson_const": ("poisson", "const"), "poisson_hourly": ("poisson", "hourly"),
                "negbin_const": ("negbin", "const"), "negbin_hourly": ("negbin", "hourly")}
N_MEAN_PARAMS = {"const": 1, "hourly": 48}
# Categorical colours, fixed order (validated palette, slots 1-4)
DIST_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

def fit_gap_dists(g_tr, g_te):
    """Fit on train-day gaps: AIC + KS on train, mean log-lik on test-day gaps."""
    out = {}
    for name, (dist, k) in GAP_DISTS.items():
        p = dist.fit(g_tr, floc=0)
        out[name] = dict(params=p,
                         aic=2 * k - 2 * dist.logpdf(g_tr, *p).sum(),
                         ho_ll=dist.logpdf(g_te, *p).mean(),
                         ks=stats.kstest(g_tr, dist.cdf, args=p).statistic)
    return out

def count_logpmf(family, k, mu, r=None):
    if family == "poisson":
        return stats.poisson.logpmf(k, mu)
    return stats.nbinom.logpmf(k, r, r / (r + mu))

def fit_nb_size(counts, mu):
    """MLE of the negative-binomial size r for a given mean path mu."""
    nll = lambda lr: -count_logpmf("negbin", counts, mu, np.exp(lr)).sum()
    return float(np.exp(optimize.minimize_scalar(nll, bounds=(-6, 12), method="bounded").x))

def fit_count_models(counts, mu):
    """mu: dict mean-model -> hourly mean path (fitted on train days). r fitted on train hours."""
    out = {}
    for name, (fam, mm) in COUNT_MODELS.items():
        k = N_MEAN_PARAMS[mm] + (fam == "negbin")
        r = fit_nb_size(counts[train], mu[mm][train]) if fam == "negbin" else None
        out[name] = dict(r=r,
                         aic=2 * k - 2 * count_logpmf(fam, counts[train], mu[mm][train], r).sum(),
                         ho_ll=count_logpmf(fam, counts[te], mu[mm][te], r).mean())
    return out

# ------------------------------------------------------------ per-station fits
rows, store, long_rates, dist_rows = {}, {}, [], {}
n_skip = 0
for name, d in df.groupby("station", observed=True):
    t = np.sort(d["t"].to_numpy())
    on_tr = train[np.floor(t).astype(int)]
    n, n_te = int(on_tr.sum()), int((~on_tr).sum())
    if n < MIN_ARRIVALS or n_te < MIN_TEST:
        n_skip += 1
        continue
    counts = np.bincount(np.floor(t).astype(int), minlength=H)[:H].astype(float)
    lam = n / train.sum()                                          # constant rate, train days

    r_tr = cell_rates(counts, train, days_in_cell_tr, lam)         # hour-of-day rates, train days
    mu_nh, mu_hom = r_tr[cell], np.full(H, lam)
    t_tr, t_te = t[on_tr], t[~on_tr]
    g_hom, g_nh = rescale(t_tr, mu_hom * train), rescale(t_tr, mu_nh * train)
    g_nh_te = rescale(t_te, mu_nh * te)

    rows[name] = dict(
        n_arrivals=n, n_test_arrivals=n_te, rate_hom_per_hr=lam,
        cv_hom=g_hom.std(ddof=1) / g_hom.mean(), cv_nhpp=g_nh.std(ddof=1) / g_nh.mean(),
        ks_hom=stats.kstest(g_hom, "expon").statistic, ks_nhpp=stats.kstest(g_nh, "expon").statistic,
        disp_hom=disp(counts[train], lam), disp_nhpp=disp(counts[train], mu_nh[train]),
        disp_ho_hom=disp(counts[te], lam), disp_ho_nhpp=disp(counts[te], mu_nh[te]))

    # which distribution fits best (fit on train days, score on test days)
    gaps = fit_gap_dists(g_nh, g_nh_te)
    cnts = fit_count_models(counts, {"const": mu_hom, "hourly": mu_nh})
    drow = {"n_arrivals": n, "n_test_arrivals": n_te}
    for kind, fits in (("gap", gaps), ("count", cnts)):
        for m, f in fits.items():
            drow[f"{kind}_aic_{m}"] = f["aic"]
            drow[f"{kind}_heldout_ll_{m}"] = f["ho_ll"]
        drow[f"best_{kind}_aic"] = min(fits, key=lambda m: fits[m]["aic"])
        drow[f"best_{kind}_heldout"] = max(fits, key=lambda m: fits[m]["ho_ll"])
    for m, f in gaps.items():
        drow[f"gap_ks_{m}"] = f["ks"]
    drow["gamma_shape"] = gaps["gamma"]["params"][0]
    drow["weibull_shape"] = gaps["weibull"]["params"][0]
    drow["negbin_size_hourly"] = cnts["negbin_hourly"]["r"]
    dist_rows[name] = drow

    store[name] = dict(g_hom=g_hom, g_nh=g_nh, g_nh_te=g_nh_te, counts=counts, mu_nh=mu_nh,
                       mu_hom=mu_hom, r=r_tr, gap_fits=gaps, count_fits=cnts)
    for c in range(48):
        long_rates.append((name, "weekend" if c >= 24 else "weekday", c % 24, r_tr[c]))
    if len(rows) % 250 == 0:
        print(f"  ... {len(rows)} stations fitted [{time.time() - T0:.0f}s]", flush=True)

res = pd.DataFrame(rows).T.sort_values("n_arrivals", ascending=False)
res.index.name = "station"
res.to_csv(f"{OUT}/station_fit_comparison.csv")
pd.DataFrame(long_rates, columns=["station", "daytype", "hour", "rate_per_hour"]).to_csv(
    f"{OUT}/station_rates_by_hour.csv", index=False)

print(f"\nStations fitted: {len(res)} ({n_skip} skipped: < {MIN_ARRIVALS} train or < {MIN_TEST} test departures)"
      f"   (medians across stations; target value in brackets)")
for lab, a, b, tgt in [("CV of rescaled gaps", "cv_hom", "cv_nhpp", 1), ("KS statistic", "ks_hom", "ks_nhpp", 0),
                       ("Count dispersion (in-sample)", "disp_hom", "disp_nhpp", 1),
                       ("Count dispersion (held-out test days)", "disp_ho_hom", "disp_ho_nhpp", 1)]:
    print(f"  {lab:38s} constant {res[a].astype(float).median():6.2f} -> hourly {res[b].astype(float).median():6.2f}   [{tgt}]")

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
    ax.set(xlim=(0, 5), title=f"{name}\n(n={int(r.n_arrivals)} train departures)", xlabel="rescaled inter-arrival (mean = 1)", ylabel="CDF")
    ax.legend(fontsize=7)

    ax = axes[1, j]                                              # Q-Q log-log
    for gaps, col in [(s["g_hom"], HOM), (s["g_nh"], NH)]:
        x = np.sort(gaps); q = stats.expon.ppf((np.arange(1, len(x) + 1) - .5) / len(x))
        ax.plot(q, x, ".", ms=3, color=col)
    ax.plot([1e-3, 10], [1e-3, 10], "r--"); ax.set(xscale="log", yscale="log", xlabel="Exp(1) quantile",
                                                    ylabel="observed quantile", title="Q-Q (log-log)")

    ax = axes[2, j]                                              # hourly count distribution
    c = s["counts"][train]; k = np.arange(0, int(c.max()) + 1)
    ax.hist(c, bins=np.arange(-0.5, c.max() + 1.5, max(1, int(c.max()) // 25)), density=True,
            color="#c9d6e8", label="observed")
    ax.plot(k, stats.poisson.pmf(k, s["mu_hom"][0]), color=HOM, marker=".", label=f"constant (disp {r.disp_hom:.1f})")
    ax.plot(k, np.mean([stats.poisson.pmf(k, m) for m in s["mu_nh"][train]], axis=0), color=NH, marker=".",
            label=f"hour-of-day (disp {r.disp_nhpp:.1f})")
    ax.set(xlabel="departures per hour", ylabel="probability", title="Hourly counts vs model (train days)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_nhpp_station_fits.png", dpi=150); plt.close()

# ----------------------------------- Fig B: before/after across all stations
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, (a, b, title, tgt, xl) in zip(axes, [
        ("cv_hom", "cv_nhpp", "Rescaled-gap CV", 1, "CV (1 = exponential)"),
        ("ks_hom", "ks_nhpp", "KS statistic on rescaled gaps", 0, "KS statistic (0 = perfect)"),
        ("disp_hom", "disp_nhpp", "Hourly-count dispersion, train days", 1, "var/mean (1 = Poisson)"),
        ("disp_ho_hom", "disp_ho_nhpp", "Hourly-count dispersion, held-out test days", 1, "var/mean (1 = Poisson)")]):
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

# ============================================ which distribution fits best?
dres = pd.DataFrame(dist_rows).T.sort_values("n_arrivals", ascending=False)
dres.index.name = "station"
dres.to_csv(f"{OUT}/station_distribution_fits.csv")

def summarize(kind, models, title):
    aic = dres[[f"{kind}_aic_{m}" for m in models]].astype(float).to_numpy()
    d_aic = aic - aic.min(axis=1, keepdims=True)
    ho = dres[[f"{kind}_heldout_ll_{m}" for m in models]].astype(float)
    print(f"\n{title}  ({len(dres)} stations)")
    print(f"  {'model':16s} {'best held-out':>13s} {'best AIC':>9s} {'median dAIC':>12s} {'median held-out LL':>19s}")
    for i, m in enumerate(models):
        print(f"  {m:16s} {(dres[f'best_{kind}_heldout'] == m).sum():>13d} "
              f"{(dres[f'best_{kind}_aic'] == m).sum():>9d} {np.median(d_aic[:, i]):>12.1f} "
              f"{ho.iloc[:, i].median():>19.4f}")
    winner = dres[f"best_{kind}_heldout"].value_counts().idxmax()
    print(f"  -> best overall (most held-out wins): {winner}")
    return winner

best_gap = summarize("gap", list(GAP_DISTS), "Inter-arrival times (rescaled by hour-of-day rate)")
best_cnt = summarize("count", list(COUNT_MODELS), "Hourly departure counts")
print(f"\n  median fitted gamma shape {dres.gamma_shape.astype(float).median():.2f}, "
      f"Weibull shape {dres.weibull_shape.astype(float).median():.2f} (1 = exponential); "
      f"median NB size r {dres.negbin_size_hourly.astype(float).median():.2f} (inf = Poisson)")

# ----------------------------- Fig D: candidate fits for the busiest stations
NB_COL = "#6250d6"
fig, axes = plt.subplots(2, N_SHOW, figsize=(4.2 * N_SHOW, 7.6))
for j, name in enumerate(show):
    s = store[name]
    ax = axes[0, j]                                   # density of log10(rescaled gap)
    u = np.log10(s["g_nh_te"])
    ax.hist(u, bins=50, density=True, color="#d9d8d3", label="observed")
    uu = np.linspace(np.percentile(u, 0.5), u.max(), 300)
    for (m, (dist, _)), col in zip(GAP_DISTS.items(), DIST_COL):
        f = s["gap_fits"][m]
        ax.plot(uu, dist.pdf(10 ** uu, *f["params"]) * 10 ** uu * np.log(10), color=col, lw=2,
                label=f"{m} (KS {f['ks']:.3f})")
    ax.set(title=f"{name}\ntest days ({len(u)} gaps), train-fitted curves", xlabel="log10 rescaled inter-arrival",
           ylabel="density")
    ax.legend(fontsize=7)

    ax = axes[1, j]                                   # hourly counts, log y to show the tail
    c = s["counts"][te]; k = np.arange(0, int(c.max()) + 1)
    ax.bar(k, np.bincount(c.astype(int)) / len(c), width=0.9, color="#d9d8d3", label="observed")
    r = s["count_fits"]["negbin_hourly"]["r"]
    ax.plot(k, np.mean([stats.poisson.pmf(k, m) for m in s["mu_nh"][te]], axis=0), color=DIST_COL[0],
            lw=2, label="Poisson, hourly mean")
    ax.plot(k, np.mean([stats.nbinom.pmf(k, r, r / (r + m)) for m in s["mu_nh"][te]], axis=0),
            color=NB_COL, lw=2, label=f"neg. binomial, hourly mean (r={r:.1f})")
    ax.set(yscale="log", ylim=(0.5 / len(c), 1), xlabel="departures per hour (test days)", ylabel="probability")
    ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_distribution_fits.png", dpi=150); plt.close()

# --------------------------- Fig E: how often each candidate wins (held-out)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, kind, models, title in [(axes[0], "gap", list(GAP_DISTS), "Inter-arrival distribution"),
                                (axes[1], "count", list(COUNT_MODELS), "Hourly-count model")]:
    wins = dres[f"best_{kind}_heldout"].value_counts().reindex(models, fill_value=0)
    ax.barh(models[::-1], wins[::-1], color=NH, height=0.6)
    for y, v in enumerate(wins[::-1]):
        ax.text(v, y, f" {v}", va="center", fontsize=9)
    ax.set(title=f"{title}: # stations where it fits best\n(test-day log-likelihood, {len(dres)} stations)",
           xlabel="# stations", xlim=(0, len(dres) * 1.1))
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_distribution_wins.png", dpi=150); plt.close()
print(f"\nDone in {time.time() - T0:.0f}s; outputs written to {OUT}/")
