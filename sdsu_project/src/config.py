"""Project-wide constants. Edit the TODO values after you look them up."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "SDSU_2026.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "bip_clean.csv"
WEATHER_CSV = ROOT / "data" / "raw" / "weather_myf.csv"   # optional, see 00_fetch_weather.py
FIG_DIR = ROOT / "reports" / "figures"

# --- Park ---------------------------------------------------------------
TGS_VENUE_ID = 711670331          # gameVenueId for Tony Gwynn Stadium in this export
PARK_LAT, PARK_LON = 32.7745, -117.0710   # approximate; plenty accurate for sun angles
PARK_TZ = "America/Los_Angeles"

# TODO: measure in Google Earth (ruler tool, home plate -> center field flagpole).
# Compass bearing in degrees clockwise from true north. Needed to express sun glare relative to the field.
FIELD_AZIMUTH_DEG = None
# TODO: fence distances (ft) at spray angles -45, -22.5, 0, 22.5, 45 (LF line .. RF line).
FENCE_FT = None

# --- Timing assumptions ---------------------------------------------------
# The export has NO per-pitch timestamps, only first pitch + pitchNumInGame.
# Elapsed time is estimated as pitchNumInGame * SEC_PER_PITCH (~3h game / ~300 pitches).
SEC_PER_PITCH = 36
# Some games carry Eastern-time stamps on the exact hour (verified: Seattle U 4/20 was 11:00 PT = 14:00 ET).
# Any gameDate with minute == 0 is treated as Eastern and shifted -3h. Verify against the schedule.
ET_TO_PT_HOURS = 3

# Light classes by sun elevation (degrees) at estimated time of the batted ball
DAY_EL = 10      # >= this: day (sun clearly up)
NIGHT_EL = -6    # <= this: night (civil dusk or later; lights on)
# between NIGHT_EL and DAY_EL = 'transition' (low sun / twilight)
