"""
Approximate batted-ball flight model, used because the export has NO landing distance or hang time
(the TruMedia fielding columns hang/dist/react/speed/jump/wall/outProb are all 0/NaN).

2-D point-mass trajectory with quadratic drag and a constant-coefficient lift term (backspin).
Cd and Cl are NOT physical measurements here: they are tuned so the model reproduces the 19 home runs in
the data that carry a reported distance (RMSE ~18 ft). They absorb spin, air density, wind, etc.

Caveats to keep in mind:
  * Fit only on home runs (EV ~94-107 mph, LA ~22-38 deg). Weak-contact balls are extrapolation.
  * Distance = where the ball first returns to ground level (no bounce/roll).
  * Hang time inherits the same uncertainty.
"""
import numpy as np
from scipy.optimize import minimize

CD_DEFAULT, CL_DEFAULT = 0.403, 0.251   # fitted on 2026 TGS home runs
G = 9.81
RHO = 1.18                              # kg/m^3 (~sea level, ~20C)
MASS, RADIUS = 0.145, 0.0366            # kg, m
FT = 0.3048


def flight(ev_mph, la_deg, cd=CD_DEFAULT, cl=CL_DEFAULT, contact_height_ft=3.0, dt=0.01):
    """Return (distance_ft, hang_time_s, apex_ft). NaN in -> NaN out."""
    if not (np.isfinite(ev_mph) and np.isfinite(la_deg)):
        return np.nan, np.nan, np.nan
    k = 0.5 * RHO * np.pi * RADIUS**2 / MASS
    v = ev_mph * 0.44704
    th = np.radians(la_deg)
    x, y = 0.0, contact_height_ft * FT
    vx, vy = v * np.cos(th), v * np.sin(th)
    t, apex = 0.0, y
    while True:
        sp = np.hypot(vx, vy)
        ax = -k * cd * sp * vx - k * cl * sp * vy
        ay = -G - k * cd * sp * vy + k * cl * sp * vx
        vx += ax * dt; vy += ay * dt
        x += vx * dt; y += vy * dt; t += dt
        apex = max(apex, y)
        if (y <= 0 and t > 0.05) or t > 15:
            break
    return x / FT, t, apex / FT


def add_flight(df, ev="ExitVel", la="LaunchAng", spray="SprayAng", **kw):
    """Add est_dist_ft, hang_s, apex_ft, land_x, land_y (home plate = origin, +x toward RF, +y toward CF)."""
    out = [flight(e, l, **kw) for e, l in zip(df[ev], df[la])]
    df = df.copy()
    df["est_dist_ft"] = [o[0] for o in out]
    df["hang_s"] = [o[1] for o in out]
    df["apex_ft"] = [o[2] for o in out]
    th = np.radians(df[spray])
    df["land_x"] = df["est_dist_ft"] * np.sin(th)
    df["land_y"] = df["est_dist_ft"] * np.cos(th)
    # the airborne model is not meaningful for grounders / very low liners
    ground = df[la] < 5
    for c in ["est_dist_ft", "hang_s", "apex_ft", "land_x", "land_y"]:
        df.loc[ground, c] = np.nan
    return df


def refit(ev, la, true_dist, x0=(0.35, 0.15)):
    """Re-tune (Cd, Cl) against any set of balls with known distance (e.g. more seasons of HRs)."""
    ev, la, true_dist = map(np.asarray, (ev, la, true_dist))

    def rmse(p):
        pred = np.array([flight(e, l, p[0], p[1])[0] for e, l in zip(ev, la)])
        return np.sqrt(np.mean((pred - true_dist) ** 2))

    r = minimize(rmse, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-2})
    return r.x, r.fun
