# Day vs. Night Batted-Ball Analysis at Tony Gwynn Stadium
 
**Question:** Do batted ball locations and outcomes differ spatially between day and night at San Diego State's home park, and can that inform outfield positioning?
 
## Status
 
| Phase | Description | State |
|---|---|---|
| 1 | Data cleaning, timestamp correction, sun-position features | Done |
| 2 | Batted ball flight model (distance, hang time) | Done |
| 3 | Day/night (single season) | Done |
| 4 | Multi-season replication | Needs more data |
| 5 | Batted ball density and out probability models | Designed |
| 6 | Outfield positioning optimizer | Designed |
 
This is an in progress analysis. The positioning optimizer is **not** built yet.
 
## Findings (2026 season, exploratory)
 
Sample: 648 balls in play across 27 home games; 428 in the day vs night comparison (219 day, 209 night); 216 fieldable outfield balls.
 
- **No statistically detectable day/night difference.** The key test, light × field location interaction on hit probability (GEE logistic regression, clustered by game), gives χ² = 2.72, df = 4, **p = 0.61**. Replacing the binary label with continuous sun elevation gives p = 0.29; adding an inning control gives p = 0.54.
- **Batted ball distribution is similar.** Night minus day differences in spray angle (−0.4°), exit velocity (−2.0 mph), and launch angle (−1.5°) all have 95% game cluster bootstrap intervals that include zero.
- **Night hit rates are higher in all three zones** (LF 67% vs. 51%, CF 39% vs. 28%, RF 38% vs. 35%), but none survive Holm correction (adjusted p ≥ 0.68).
- **The sample is underpowered.** At 80% power, the smallest detectable day/night hit-rate gap is roughly 29 to 35 percentage points per zone. A null result here is not evidence of no effect. Replication across additional seasons is the planned fix.
![Hit rate by zone](reports/figures/fig2_hit_rate_by_zone.png)
 
## Methods
 
- **Day/night from sun position, not game labels.** 16 of 27 games start in daylight and end after dark, so labeling whole games would misclassify most of the sample. Each ball is classified by computed sun elevation at its estimated time (day ≥ 10°, night ≤ −6°, transition band excluded from the binary contrast). The result is checked against alternate assumptions about game pace.
- **Timestamp correction.** Ten games carried Eastern time first pitch stamps in a Pacific time dataset so they were detected from the schedule pattern and shifted. One game flagged as a home game was played at a different venue and was excluded.
- **Missing distance and hang time.** The source fielding columns are empty, so a drag plus lift trajectory model estimates landing spot and hang time from exit velocity, launch angle, and spray angle. It was tuned on 19 home runs with reported distances (RMSE ≈ 18 ft, bias −0.3 ft). It was tuned only on home runs, so estimates for weak contact are extrapolations.
- **Cluster aware inference.** Balls from the same game are not independent. Tests use GEE with exchangeable correlation by game, and mean differences use game level cluster bootstrap CIs.
- **Home runs analyzed separately.** They are unfieldable and reflect carry, not positioning.
## Data
 
Columns the pipeline uses:
 
| Purpose | Columns |
|---|---|
| Location | `SprayAng` |
| Contact quality | `ExitVel`, `LaunchAng` |
| Outcome | `ABResult`, `pitchResult`, `atbatDesc` |
| Context | `gameId`, `gameDate`, `gameVenueId`, `pitchNumInGame`, `inn`, `batterHand`, `batterId`, `pitcherId`, `opponent` |
 
## Quick start
 
```bash
pip install -r requirements.txt
python src/01_prepare_data.py       # raw CSV -> data/processed/bip_clean.csv
python src/02_day_night_pilot.py    # statistics + figures -> reports/
```
 
## Project structure
 
```
src/
  config.py              constants and assumptions (pace, light thresholds, park)
  physics.py             batted ball flight model
  00_fetch_weather.py    optional weather download
  01_prepare_data.py     cleaning and feature engineering
  02_day_night_pilot.py  statistical tests and figures
reports/                 figures, data quality log, pilot results
```
 
## Limitations
 
- One season. The study is underpowered (see Findings).
- No per pitch timestamps exist, so time of each ball is estimated from first pitch and pitch count (36 s/pitch assumed). Changing this assumption shifts the share of balls classified as night from roughly 31% to 45%.
- Landing distance and hang time are model estimates, not measurements.
- No fielder starting positions are available, so the analysis cannot separate "hard to reach ball" from "fielder out of position."
- Weather is not yet included in the models and is a likely confounder.
## Planned work
 
1. Add 2023 to 2025 seasons for confirmatory testing, with the analysis plan fixed in advance.
2. Merge hourly weather and move to mixed effects models (random effects for game, batter, pitcher).
3. Fit batted ball density and out probability models with leave one game out validation.
4. Build a kinematic reachability model and positioning optimizer with bootstrap uncertainty.
