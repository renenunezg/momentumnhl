from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "backend" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR = REPO_ROOT / "backend" / "data_static"

MODEL_VERSION = "nhl-poisson-v1.1"
POLICY_VERSION = "nhl-picks-v1"

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
MONEYLINE_MIN_EDGE = 0.13
TOTAL_MIN_EDGE_GOALS = 1.0
KELLY_FRACTION = 0.10
MIN_PRICE_MULTIPLIER = 1.1
# Partner feeds carry one timestamp for the whole slate; anything older than
# this at decision time is not a live quote.
MAX_OFFER_AGE_HOURS = 24

SITE_TIME_ZONE = "America/Los_Angeles"
