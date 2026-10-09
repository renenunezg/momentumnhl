from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "backend" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR = REPO_ROOT / "backend" / "data_static"

MODEL_VERSION = "nhl-poisson-v1.2"
POLICY_VERSION = "nhl-picks-v2"

# Seasons are start years: 2026 is 2026-27. MoneyPuck rows exist from 2008; the
# model only needs a rolling window plus a few seasons for the goal map and
# the backtest.
HISTORY_START_SEASON = 2021
GOAL_MAP_SEASONS = (2021, 2022, 2023, 2024)
BACKTEST_SEASONS = (2022, 2023, 2024, 2025)

# Sheet parameters, kept as the workbook had them.
WINDOW_GAMES = 25
MIN_WINDOW_GAMES = 10
GRID_MAX_GOALS = 10
KELLY_FRACTION = 0.10
MIN_PRICE_MULTIPLIER = 1.1

# A game the books quote publishes a market-anchored forecast, as the MLB and
# NFL models do, and picks price that same forecast: the win probability
# blends with the de-vigged moneyline and the goal total with the posted
# total. Against 2022 to 2025 open and closing lines the pure model added
# nothing to either price (the fitted model weight is near 0) and the sheet's
# gates lost at every setting. The weight and the gates are a product decision
# so picks still surface, not fitted estimates; the backtest shows no edge at
# these values. Do not retune them without Rene's approval.
MARKET_ANCHOR_W_MODEL = 0.5
MONEYLINE_MIN_EDGE = 0.045
# Half a goal on the blended total is the sheet's one-goal rule on the pure
# total, because the blend halves the model's gap to the line.
TOTAL_MIN_EDGE_GOALS = 0.5
# The pure model under-rates home teams (51.5 percent predicted, 53.8 actual);
# this logit shift was fitted on the 5,212 walk-forward backtest games.
HOME_ICE_LOGIT = 0.10
# Partner feeds carry one timestamp for the whole slate; anything older than
# this needs independent public-listing corroboration before it is eligible.
MAX_OFFER_AGE_HOURS = 24

SITE_TIME_ZONE = "America/Los_Angeles"
