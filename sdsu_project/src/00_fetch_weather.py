"""
Optional Step 0b: download hourly weather for the 2026 season from the Iowa Environmental Mesonet (free, no key).
Station MYF = Montgomery-Gibbs Executive Airport, the closest ASOS site to SDSU. KSAN (Lindbergh) is the alternate.

NOTE: this script could not be test-run where it was written (no outbound network). If IEM changes its API,
check https://mesonet.agron.iastate.edu/request/download.phtml for the current parameters.

Run:  python src/00_fetch_weather.py          (needs internet)
Out:  data/raw/weather_myf.csv  -> picked up automatically by 01_prepare_data.py
"""
import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
params = {
    "station": "MYF",
    "data": ["tmpf", "relh", "sknt", "drct", "alti"],   # temp F, rel. humidity, wind kt, wind dir, altimeter
    "year1": 2026, "month1": 2, "day1": 12,
    "year2": 2026, "month2": 5, "day2": 12,
    "tz": C.PARK_TZ, "format": "onlycomma", "latlon": "no",
    "missing": "M", "trace": "T", "direct": "no",
    "report_type": [3, 4],
}
r = requests.get(URL, params=params, timeout=60)
r.raise_for_status()
w = pd.read_csv(io.StringIO(r.text), na_values=["M"])
w.to_csv(C.WEATHER_CSV, index=False)
print(f"saved {len(w)} observations to {C.WEATHER_CSV}")
