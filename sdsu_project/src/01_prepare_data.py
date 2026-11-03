"""
Step 1: raw TruMedia/TrackMan export -> one clean row per ball in play at Tony Gwynn Stadium.

Run:  python src/01_prepare_data.py
Out:  data/processed/bip_clean.csv  and  reports/data_quality.txt
"""
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import azimuth, elevation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from physics import add_flight, flight, refit

PT = ZoneInfo(C.PARK_TZ)
OBS = LocationInfo("TGS", "USA", C.PARK_TZ, C.PARK_LAT, C.PARK_LON).observer
HIT_BASES = {"S": 1, "D": 2, "T": 3, "HR": 4}
log_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log_lines.append(s)


def num(s):
    """TruMedia uses '-' for missing."""
    return pd.to_numeric(s, errors="coerce")


def parse_bb_type(txt):
    t = str(txt)
    if "Bunt" in t or "Sac Bunt" in t:
        return "bunt"
    if "Ground" in t or "Double Play" in t or "Fielder's Choice" in t:
        return "ground"
    if "Line" in t:
        return "line"
    if "Pop" in t:
        return "popup"
    if "Fly" in t:          # includes "Sac Fly"
        return "fly"
    return "other"


def parse_fielder(desc):
    """First fielder credited in the scorebook string. '8/F'->8, '43/G'->4, 'S/7'->7, 'D/78'->7, 'FC6.1X2'->6."""
    d = str(desc)
    m = re.match(r"^(?:FC)?([1-9])", d)
    if m:
        return int(m.group(1))
    m = re.match(r"^(?:S|D|T|HR)/([1-9])", d)
    if m:
        return int(m.group(1))
    m = re.match(r"^E([1-9])", d)
    return int(m.group(1)) if m else np.nan


def sun_at(ts):
    t = ts.to_pydatetime().replace(tzinfo=PT)
    return elevation(OBS, t), azimuth(OBS, t)


def main():
    raw = pd.read_csv(C.RAW_CSV)
    log(f"raw rows: {len(raw)}, games: {raw.gameId.nunique()}")

    # ---- 1. venue filter -------------------------------------------------
    off = raw[raw.gameVenueId != C.TGS_VENUE_ID]
    log(f"dropping {len(off)} rows from {off.gameId.nunique()} game(s) not at Tony Gwynn Stadium: "
        f"{off.opponent.unique().tolist()} (flagged home=True in export, different venue id)")
    df = raw[raw.gameVenueId == C.TGS_VENUE_ID].copy()

    # ---- 2. numeric coercion ----------------------------------------------
    for c in ["ExitVel", "LaunchAng", "SprayAng", "ExitDir", "xAVG", "xWOBA", "xSLG", "releaseVelocity"]:
        df[c] = num(df[c])
    df["batter_hand"] = df["batterHand"]
    df["pitcher_hand"] = df["pitcherHand"]
    df["inn_n"] = df.inn.str.extract(r"(\d+)")[0].astype(int)

    # ---- 3. first-pitch time -> Pacific --------------------------------------
    gd = pd.to_datetime(df.gameDate)
    is_et = gd.dt.minute == 0
    df["first_pitch_pt"] = gd.where(~is_et, gd - pd.Timedelta(hours=C.ET_TO_PT_HOURS))
    df["fp_was_shifted"] = is_et
    log(f"first-pitch timestamps shifted ET->PT: {df.loc[is_et, 'gameId'].nunique()} games")

    # estimated clock time of the batted ball (no real per-pitch timestamp exists)
    df["t_hit_pt"] = df.first_pitch_pt + pd.to_timedelta(df.pitchNumInGame * C.SEC_PER_PITCH, unit="s")
    sa = df.t_hit_pt.map(sun_at)
    df["sun_el"] = [s[0] for s in sa]
    df["sun_az"] = [s[1] for s in sa]
    df["light"] = np.select([df.sun_el >= C.DAY_EL, df.sun_el <= C.NIGHT_EL], ["day", "night"], "transition")
    gt = df.groupby("gameId").sun_el.apply(
        lambda s: "all_day" if s.min() > 0 else ("all_night" if s.max() < 0 else "mixed"))
    df["game_type"] = df.gameId.map(gt)
    df["dow"] = df.first_pitch_pt.dt.day_name()
    df["month"] = df.first_pitch_pt.dt.month
    df["game_frac"] = df.pitchNumInGame / df.groupby("gameId").pitchNumInGame.transform("max")

    # ---- 4. outcomes ---------------------------------------------------------------
    df["result"] = df.ABResult
    df["is_hit"] = df.ABResult.isin(HIT_BASES)
    df["total_bases"] = df.ABResult.map(HIT_BASES).fillna(0).astype(int)
    df["is_error"] = df.ABResult.eq("ROE")
    df["is_sac_bunt"] = df.ABResult.eq("SH")
    df["is_out"] = ~df.is_hit & ~df.is_error
    df["bb_type"] = df.pitchResult.map(parse_bb_type)
    df["fielder"] = df.atbatDesc.map(parse_fielder)
    df["foul_out"] = df.atbatDesc.astype(str).str.contains("/PF")
    # rows suitable for "did the defense convert this ball?" modeling
    df["conv_ok"] = ~df.is_error & ~df.is_sac_bunt & ~df.foul_out & (df.bb_type != "bunt")

    # ---- 5. spatial -------------------------------------------------------------------
    df["spray_fair"] = df.SprayAng.abs() <= 45
    # pull-signed spray: + = pulled, - = opposite field (RHB pulls to LF, i.e. negative SprayAng)
    df["pull_ang"] = np.where(df.batter_hand == "R", -df.SprayAng, df.SprayAng)
    df["zone3"] = pd.cut(df.SprayAng, [-45.001, -15, 15, 45.001], labels=["LF", "CF", "RF"])
    df["zone5"] = pd.cut(df.SprayAng, [-45.001, -27, -9, 9, 27, 45.001], labels=["LF", "LCF", "CF", "RCF", "RF"])

    # physics-based landing spot, hang time (see physics.py for caveats)
    df = add_flight(df)
    # sanity check against home runs with reported distance
    hr_true = df.pitchResult.str.extract(r"([\d.]+) ft")[0].astype(float)
    m = hr_true.notna() & df.est_dist_ft.notna()
    if m.sum():
        err = df.est_dist_ft[m] - hr_true[m]
        log(f"physics check vs {m.sum()} reported HR distances: bias {err.mean():+.1f} ft, RMSE {np.sqrt((err**2).mean()):.1f} ft")
    df["hr_reported_ft"] = hr_true

    # outfield ball = credited to an OF fielder, or airborne ball with a long estimated carry
    df["of_ball"] = df.fielder.isin([7, 8, 9]) | (df.bb_type.isin(["fly", "line"]) & (df.est_dist_ft >= 200))
    df["spatial_ok"] = df.SprayAng.notna() & df.ExitVel.notna() & df.LaunchAng.notna() & df.spray_fair & ~df.foul_out

    # ---- 6. optional weather -----------------------------------------------------------
    if C.WEATHER_CSV.exists():
        w = pd.read_csv(C.WEATHER_CSV, parse_dates=["valid"]).sort_values("valid")
        for c in ["tmpf", "relh", "sknt", "drct", "alti"]:
            w[c] = pd.to_numeric(w[c], errors="coerce")
        df = df.sort_values("t_hit_pt")
        df["t_hit_naive"] = df.t_hit_pt.dt.tz_localize(None)
        w["valid"] = w["valid"].dt.tz_localize(None) if w["valid"].dt.tz is not None else w["valid"]
        df = pd.merge_asof(df, w[["valid", "tmpf", "relh", "sknt", "drct", "alti"]], left_on="t_hit_naive",
                           right_on="valid", direction="nearest", tolerance=pd.Timedelta("60min"))
        log("weather merged; non-null temp:", int(df.tmpf.notna().sum()))
    else:
        log("no weather file found (data/raw/weather_myf.csv) -> weather columns skipped")

    # ---- 7. keep tidy columns + write ----------------------------------------------------
    keep = ["gameId", "gameString", "opponent", "first_pitch_pt", "fp_was_shifted", "t_hit_pt", "dow", "month",
            "game_type", "game_frac", "pitchNumInGame", "inn_n", "outs", "count",
            "batter", "batterId", "batter_hand", "pitcher", "pitcherId", "pitcher_hand", "pitchType", "releaseVelocity",
            "pitchResult", "atbatDesc", "result", "is_hit", "total_bases", "is_error", "is_sac_bunt", "is_out",
            "bb_type", "fielder", "foul_out", "conv_ok",
            "ExitVel", "LaunchAng", "SprayAng", "ExitDir", "pull_ang", "spray_fair", "zone3", "zone5",
            "est_dist_ft", "hang_s", "apex_ft", "land_x", "land_y", "hr_reported_ft", "of_ball", "spatial_ok",
            "sun_el", "sun_az", "light", "xAVG", "xSLG", "xWOBA"]
    keep += [c for c in ["tmpf", "relh", "sknt", "drct", "alti"] if c in df.columns]
    out = df.sort_values(["first_pitch_pt", "pitchNumInGame"])[keep]
    C.CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.CLEAN_CSV, index=False)

    # ---- 8. data-quality report -----------------------------------------------------------
    log("\n=== DATA QUALITY ===")
    log(f"clean rows: {len(out)} | games: {out.gameId.nunique()}")
    log(f"missing SprayAng: {out.SprayAng.isna().sum()} | missing ExitVel/LaunchAng: {out.ExitVel.isna().sum()}")
    log(f"non-fair spray (|angle|>45): {(~out.spray_fair).sum()} | foul outs: {out.foul_out.sum()}")
    log(f"spatial_ok rows: {out.spatial_ok.sum()} | outfield balls (spatial_ok): {(out.of_ball & out.spatial_ok).sum()}")
    log("light class counts:\n" + out.light.value_counts().to_string())
    log("game types:\n" + out.drop_duplicates('gameId').game_type.value_counts().to_string())
    log("results:\n" + out.result.value_counts().to_string())
    (C.ROOT / "reports").mkdir(exist_ok=True)
    (C.ROOT / "reports" / "data_quality.txt").write_text("\n".join(log_lines))
    log(f"\nwrote {C.CLEAN_CSV}")


if __name__ == "__main__":
    main()
