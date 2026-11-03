"""
Steps 2-3: exploratory + inferential day-vs-night comparison on the 2026 TGS balls in play.

Run:  python src/02_day_night_pilot.py
Out:  reports/pilot_results.txt and reports/figures/*.png

Design choices (see roadmap doc for the reasoning):
  * 'Day' / 'night' come from SUN ELEVATION at the estimated time of each ball, not from the game label,
    because most games start in daylight and end after dark.
  * Binary comparison uses day (>=10 deg) vs night (<=-6 deg) and drops the transition band; the continuous
    sun-elevation version is run as a sensitivity check.
  * Every test is cluster-aware (games are the unit of dependence): cluster bootstrap for means,
    GEE with exchangeable correlation by game for regressions.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def cluster_boot_diff(df, col, flag="night", cluster="gameId", B=2000):
    """Difference in mean(col) night - day with a game-level (cluster) bootstrap CI."""
    games = df[cluster].unique()
    grp = {g: df[df[cluster] == g] for g in games}
    est = df.loc[df[flag] == 1, col].mean() - df.loc[df[flag] == 0, col].mean()
    boots = []
    for _ in range(B):
        samp = pd.concat([grp[g] for g in rng.choice(games, len(games))])
        a, b = samp.loc[samp[flag] == 1, col], samp.loc[samp[flag] == 0, col]
        if len(a) and len(b):
            boots.append(a.mean() - b.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, lo, hi


def wald_interaction(res, token="night"):
    ev = np.linalg.eigvalsh(np.asarray(res.cov_params()))
    assert ev.min() > -1e-8 * max(1.0, ev.max()), "covariance not PSD: check for collinear terms"
    names = list(res.params.index)
    inter = [n for n in names if token in n and ":" in n]
    R = np.zeros((len(inter), len(names)))
    for i, n in enumerate(inter):
        R[i, names.index(n)] = 1
    w = res.wald_test(R, scalar=True)
    return float(w.statistic), len(inter), float(w.pvalue)


def main():
    df = pd.read_csv(C.CLEAN_CSV, parse_dates=["first_pitch_pt", "t_hit_pt"])
    base = df[df.spatial_ok & df.conv_ok].copy()
    base["night"] = (base.light == "night").astype(int)
    base["is_hit"] = base.is_hit.astype(int)
    dn = base[base.light.isin(["day", "night"])].copy()           # binary contrast, transition dropped
    of = dn[dn.of_ball & (dn.result != "HR")].copy()   # home runs are unfieldable: analyzed separately (carry), not positioning
    hr = dn[dn.result == "HR"]
    out("=" * 70)
    out("SAMPLE")
    out(f"balls in play at TGS (all): {len(df)} | modeled set (spatial_ok & conv_ok): {len(base)}")
    out(f"binary day/night set: {len(dn)} balls ({(dn.night==0).sum()} day, {(dn.night==1).sum()} night) from {dn.gameId.nunique()} games")
    out(f"outfield-ball subset: {len(of)} ({(of.night==0).sum()} day, {(of.night==1).sum()} night)")
    out(f"home runs excluded from fieldable set: {len(hr)} ({(hr.night==0).sum()} day, {(hr.night==1).sum()} night)")
    out(f"hit rate: all {dn.is_hit.mean():.3f} | fieldable OF balls {of.is_hit.mean():.3f}")

    # ------------------------------------------------------------------ A. WHERE do balls go / how hard
    out("\n" + "=" * 70)
    out("A. DISTRIBUTION OF BATTED BALLS (night minus day, 95% cluster-bootstrap CI over games)")
    for col, label in [("SprayAng", "spray angle (deg, +RF)"), ("pull_ang", "pull-signed angle (deg, +pull)"),
                       ("ExitVel", "exit velo (mph)"), ("LaunchAng", "launch angle (deg)")]:
        e, lo, hi = cluster_boot_diff(dn, col)
        out(f"  {label:34s} diff {e:+6.2f}  CI [{lo:+6.2f}, {hi:+6.2f}]  {'<-- excludes 0' if lo > 0 or hi < 0 else ''}")
    e, lo, hi = cluster_boot_diff(of, "est_dist_ft")
    out(f"  {'OF-ball est. carry (ft)':34s} diff {e:+6.2f}  CI [{lo:+6.2f}, {hi:+6.2f}]  {'<-- excludes 0' if lo > 0 or hi < 0 else ''}")
    ks = stats.ks_2samp(dn.loc[dn.night == 0, "SprayAng"], dn.loc[dn.night == 1, "SprayAng"])
    out(f"  KS test on spray-angle distribution (ignores clustering, so optimistic): D={ks.statistic:.3f}, p={ks.pvalue:.3f}")

    # ------------------------------------------------------------------ B. WHAT happens: zone table
    out("\n" + "=" * 70)
    out("B. HIT RATE BY ZONE x LIGHT  (outfield balls; Fisher exact per zone, Holm-adjusted)")
    rows, pvals = [], []
    for z in ["LF", "CF", "RF"]:
        s = of[of.zone3 == z]
        a, b = s[s.night == 0], s[s.night == 1]
        tab = [[a.is_hit.sum(), len(a) - a.is_hit.sum()], [b.is_hit.sum(), len(b) - b.is_hit.sum()]]
        p = stats.fisher_exact(tab)[1] if len(a) and len(b) else np.nan
        ci_a = proportion_confint(a.is_hit.sum(), len(a), method="wilson") if len(a) else (np.nan,) * 2
        ci_b = proportion_confint(b.is_hit.sum(), len(b), method="wilson") if len(b) else (np.nan,) * 2
        rows.append(dict(zone=z, n_day=len(a), hit_day=a.is_hit.mean(), lo_day=ci_a[0], hi_day=ci_a[1],
                         n_night=len(b), hit_night=b.is_hit.mean(), lo_night=ci_b[0], hi_night=ci_b[1], p=p))
        pvals.append(p)
    zt = pd.DataFrame(rows)
    zt["p_holm"] = multipletests(pvals, method="holm")[1]
    out(zt.round(3).to_string(index=False))
    zone_tab = zt

    # ------------------------------------------------------------------ C. Regression with interaction
    out("\n" + "=" * 70)
    out("C. DOES THE LIGHT EFFECT VARY ACROSS THE FIELD?  (logistic, outfield balls)")
    out("   P(hit) ~ spline(spray) + exit velo + launch angle  [+ night]  [+ night x spline(spray)]")
    f0 = "is_hit ~ cr(SprayAng, df=4, constraints='center') + ExitVel + LaunchAng"
    f1 = f0 + " + night"
    f2 = "is_hit ~ cr(SprayAng, df=4, constraints='center')*night + ExitVel + LaunchAng"
    ols = {k: smf.logit(f, of).fit(disp=0) for k, f in [("M0", f0), ("M1", f1), ("M2", f2)]}
    out(f"  Naive LRT, main effect of night (M1 vs M0): chi2={2*(ols['M1'].llf-ols['M0'].llf):.2f}, "
        f"p={stats.chi2.sf(2*(ols['M1'].llf-ols['M0'].llf), 1):.3f}")
    k = len(ols['M2'].params) - len(ols['M1'].params)
    out(f"  Naive LRT, night x location interaction (M2 vs M1): chi2={2*(ols['M2'].llf-ols['M1'].llf):.2f}, df={k}, "
        f"p={stats.chi2.sf(2*(ols['M2'].llf-ols['M1'].llf), k):.3f}")
    gee = {}
    for k_, f in [("M1", f1), ("M2", f2)]:
        gee[k_] = smf.gee(f, "gameId", of, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    z, p = gee["M1"].tvalues["night"], gee["M1"].pvalues["night"]
    out(f"  GEE (clustered by game) night main effect: log-odds {gee['M1'].params['night']:+.2f} "
        f"(OR {np.exp(gee['M1'].params['night']):.2f}), p={p:.3f}")
    st, df_, pw = wald_interaction(gee["M2"])
    out(f"  GEE Wald test, night x location interaction: chi2={st:.2f}, df={df_}, p={pw:.3f}   <-- the key test")

    # continuous sensitivity
    of["sun10"] = of.sun_el / 10
    gs = smf.gee("is_hit ~ cr(SprayAng, df=4, constraints='center')*sun10 + ExitVel + LaunchAng", "gameId", of,
                 family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    st, df_, pw = wald_interaction(gs, token="sun10")
    out(f"  Sensitivity, continuous sun elevation x location (GEE Wald): chi2={st:.2f}, df={df_}, p={pw:.3f}")

    # control for inning (pitcher turnover/fatigue is confounded with light inside mixed games)
    gi = smf.gee("is_hit ~ cr(SprayAng, df=4, constraints='center')*night + ExitVel + LaunchAng + inn_n", "gameId", of,
                 family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    st, df_, pw = wald_interaction(gi)
    out(f"  Sensitivity, adding inning control: interaction chi2={st:.2f}, df={df_}, p={pw:.3f}")

    # ------------------------------------------------------------------ D. Power
    out("\n" + "=" * 70)
    out("D. POWER: smallest day-vs-night hit-rate gap detectable (80% power, alpha=.05, two-sided) per zone")
    pw_rows = []
    for z in ["LF", "CF", "RF"]:
        s = of[of.zone3 == z]
        n1, n2 = (s.night == 0).sum(), (s.night == 1).sum()
        p1 = s.is_hit.mean()
        if min(n1, n2) < 5 or not (0 < p1 < 1):
            continue
        h = NormalIndPower().solve_power(effect_size=None, nobs1=n1, ratio=n2 / n1, alpha=0.05, power=0.8)
        p2 = np.sin((2 * np.arcsin(np.sqrt(p1)) + h) / 2) ** 2
        pw_rows.append(dict(zone=z, n_day=n1, n_night=n2, base_hit_rate=round(p1, 3),
                            detectable_gap_pts=round(100 * (p2 - p1), 1)))
    out(pd.DataFrame(pw_rows).to_string(index=False))
    out("   (gaps are before clustering; true detectable gaps are larger.)")

    # ------------------------------------------------------------------ E. Figures
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Fig 1: spray charts
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, (lab, sub) in zip(axs, [("Day (sun >= 10°)", of[of.night == 0]), ("Night (sun <= -6°)", of[of.night == 1])]):
        for ang in (-45, 45):
            ax.plot([0, 330 * np.sin(np.radians(ang))], [0, 330 * np.cos(np.radians(ang))], color="gray", lw=1)
        h = sub[sub.is_hit == 1]; o = sub[sub.is_hit == 0]
        ax.scatter(o.land_x, o.land_y, s=22, c="#1f77b4", alpha=.6, label=f"out ({len(o)})")
        ax.scatter(h.land_x, h.land_y, s=30, c="#d62728", marker="^", alpha=.8, label=f"hit ({len(h)})")
        ax.set_title(f"{lab}: n={len(sub)}, hit rate {sub.is_hit.mean():.2f}")
        ax.set_aspect("equal"); ax.legend(loc="lower center"); ax.set_xlabel("ft toward RF")
    axs[0].set_ylabel("ft toward CF")
    fig.suptitle("Estimated landing spots, outfield balls at Tony Gwynn Stadium (2026). Distance is physics-modeled.")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig1_spray_by_light.png", dpi=130); plt.close(fig)

    # Fig 2: hit rate by zone with Wilson CIs
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3); w = .36
    for off_, (lab, c, col) in zip([-w / 2, w / 2], [("Day", "hit_day", "#f0a30a"), ("Night", "hit_night", "#2b3a67")]):
        lo, hi = zone_tab[f"lo_{lab.lower()}"], zone_tab[f"hi_{lab.lower()}"]
        ax.bar(x + off_, zone_tab[c], w, color=col, label=lab,
               yerr=[zone_tab[c] - lo, hi - zone_tab[c]], capsize=4)
    for i, r in zone_tab.iterrows():
        ax.text(i - w / 2, 0.02, f"n={int(r.n_day)}", ha="center", color="white", fontsize=9)
        ax.text(i + w / 2, 0.02, f"n={int(r.n_night)}", ha="center", color="white", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(zone_tab.zone); ax.set_ylabel("hit rate"); ax.set_ylim(0, 1)
    ax.set_title("Outfield-ball hit rate by zone and light (95% Wilson CI)"); ax.legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig2_hit_rate_by_zone.png", dpi=130); plt.close(fig)

    # Fig 3: spray-angle densities
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, ttl in [(axs[0], "SprayAng", "Spray angle (+ = RF)"), (axs[1], "pull_ang", "Pull-signed angle (+ = pulled)")]:
        for flag, lab, colr in [(0, "Day", "#f0a30a"), (1, "Night", "#2b3a67")]:
            v = dn.loc[dn.night == flag, col].dropna()
            kde = stats.gaussian_kde(v)
            xs = np.linspace(-45, 45, 200)
            ax.plot(xs, kde(xs), color=colr, label=f"{lab} (n={len(v)})")
            ax.fill_between(xs, kde(xs), alpha=.15, color=colr)
        ax.set_title(ttl); ax.set_xlabel("degrees"); ax.legend()
    fig.suptitle("All fair balls in play: where they go, day vs night")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig3_spray_density.png", dpi=130); plt.close(fig)

    out("\nfigures saved to reports/figures/")
    (C.ROOT / "reports" / "pilot_results.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
