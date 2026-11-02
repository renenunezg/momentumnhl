from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "backend" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR = REPO_ROOT / "backend" / "data_static"

MODEL_VERSION = "nhl-poisson-v1.3"
POLICY_VERSION = "nhl-picks-v3"

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
# NFL models do, and picks price that same forecast. Against 2023 to 2025
# ESPN open lines the pure model adds nothing to the de-vigged moneyline (its
# fitted logit weight is 0.02 +/- 0.07), so the win probability is the
# consensus itself, moved only when one team played the day before. The open
# under-prices that: on 3,890 games a back-to-back costs 0.19 +/- 0.075 logit
# beyond the open (0.13 beyond the close). Walk-forward on 2024 and 2025 at
# the open, picking every side with positive expected value went 325-230,
# +5.3 +/- 4.0 percent, hitting 58.6 percent against 58.4 expected.
BACK_TO_BACK_LOGIT = -0.19
# The goal total still blends halfway with the posted total; that weight and
# the half-goal gate are Rene's product decision, not fitted estimates, and
# the backtest shows no edge for totals at any setting. Do not retune them
# without Rene's approval.
MARKET_ANCHOR_W_MODEL = 0.5
# Half a goal on the blended total is the sheet's one-goal rule on the pure
# total, because the blend halves the model's gap to the line. Since v2.1 a
# total also needs positive expected value at its price, as a moneyline does.
TOTAL_MIN_EDGE_GOALS = 0.5
# Partner feeds carry one timestamp for the whole slate; anything older than
# this needs independent public-listing corroboration before it is eligible.
MAX_OFFER_AGE_HOURS = 24

# A partner quote is a closing line only when the feed stamped it this close
# to the scheduled start: three of FanDuel's five-minute restamps. The live
# worker selects the close here and writes one row to nhl.closing_lines.
CLOSING_LINE_MINUTES = 15

SITE_TIME_ZONE = "America/Los_Angeles"
