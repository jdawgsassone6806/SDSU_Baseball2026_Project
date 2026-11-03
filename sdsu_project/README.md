# SDSU / Tony Gwynn Stadium: day vs. night batted-ball analysis

    pip install -r requirements.txt
    python src/00_fetch_weather.py      # optional, needs internet (weather confounder)
    python src/01_prepare_data.py       # raw CSV -> data/processed/bip_clean.csv + reports/data_quality.txt
    python src/02_day_night_pilot.py    # pilot stats + figures -> reports/

Put new season exports in data/raw/ and point `RAW_CSV` in src/config.py at them (or concatenate first).
Fill in the TODOs in src/config.py (field azimuth, fence distances) before the glare and optimizer steps.
See the roadmap document for the full plan.
