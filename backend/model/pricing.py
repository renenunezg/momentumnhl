"""Odds arithmetic exactly as the workbook computes it, plus the market
anchors a quoted game's published win probability and total move to."""

import math

from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import poisson

from backend.config import BACK_TO_BACK_LOGIT, KELLY_FRACTION, MIN_PRICE_MULTIPLIER


def american_to_decimal(price: float) -> float:
    return price / 100 + 1 if price > 0 else 100 / abs(price) + 1


def decimal_to_american(decimal: float) -> int:
    american = (decimal - 1) * 100 if decimal >= 2 else -100 / (decimal - 1)
    rounded = int(round(american))
    # Both signs mean even money; use one representation across replay roundoff.
    return 100 if rounded == -100 else rounded


def profit_per_unit(price: float) -> float:
    return american_to_decimal(price) - 1


def implied_probability(price: float) -> float:
    return 1 / american_to_decimal(price)


def devig(probability_a: float, probability_b: float) -> tuple[float, float]:
    """Two-way implied probabilities scaled to sum to one."""
    total = probability_a + probability_b
    return probability_a / total, probability_b / total


def anchored_home_probability(market_home: float, back_to_back: int) -> float:
    """The de-vigged market consensus, shifted in logit space when one team
    played the day before: `back_to_back` is the home team's flag minus the
    away team's."""
    return float(expit(logit(market_home) + BACK_TO_BACK_LOGIT * back_to_back))


def implied_total(line: float, over_price: float, under_price: float) -> float:
    """The mean goal total at which a Poisson total gives the de-vigged over
    probability, so the juice moves the market's total off its line. A push
    is excluded, as a two-way price excludes it."""
    over = devig(implied_probability(over_price), implied_probability(under_price))[0]

    def gap(mean: float) -> float:
        above = poisson.sf(math.floor(line), mean)
        below = poisson.cdf(math.ceil(line) - 1, mean)
        return above / (above + below) - over

    return float(brentq(gap, 0.05, 30.0, xtol=1e-12))


def kelly_fraction(
    probability: float, decimal: float, fraction: float = KELLY_FRACTION
) -> float:
    """Fraction of bankroll to stake: the sheet's fractional Kelly, never
    negative."""
    full = (probability * decimal - 1) / (decimal - 1)
    return max(0.0, full * fraction)


def expected_value_per_unit(
    probability: float, price: float, push: float = 0.0
) -> float:
    return probability * profit_per_unit(price) - (1 - probability - push)


def probability_edge(probability: float, price: float, push: float = 0.0) -> float:
    """Model probability above the price's break-even probability. With no
    push this is the sheet's implied-probability difference."""
    return probability / (1 - push) - 1 / (1 + profit_per_unit(price))


def minimum_price(decimal: float, multiplier: float = MIN_PRICE_MULTIPLIER) -> int:
    """The sheet's 'Play At' column: the fair decimal marked up by 10 percent."""
    return decimal_to_american(decimal * multiplier)
