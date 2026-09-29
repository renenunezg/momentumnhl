"""Rolling goal/xG strengths with explicit league shrinkage and venue pooling.

All joins require a strictly earlier game date. MoneyPuck xG is the provider's
current historical revision, not a point-in-time archive of its xG model.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import skellam


@dataclass(frozen=True)
class Specification:
    xg_weight: float = 1.0
    prior_games: int = 25
    venue_weight: float = 0.5
    overtime: bool = False
    goalie_weight: float = 0.0

    def __post_init__(self):
        if (
            not all(
                0 <= w <= 1
                for w in (self.xg_weight, self.venue_weight, self.goalie_weight)
            )
            or self.prior_games < 0
        ):
            raise ValueError("Weights must be in [0,1] and prior games nonnegative")

    @property
    def name(self):
        return (
            f"xg{self.xg_weight:g}-prior{self.prior_games}"
            f"-venue{self.venue_weight:g}-ot{int(self.overtime)}"
            f"-goalie{self.goalie_weight:g}"
        )


SIGNALS = ["gf", "ga", "xgf", "xga", "reg_gf", "reg_ga", "reg_xgf", "reg_xga"]


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "game_date", "game_id", "team_abbr", "venue"]
    rows = games.groupby(keys, as_index=False)[["gf", "ga", "xgf", "xga", "toi"]].sum()
    if rows.duplicated(["game_id", "team_abbr"]).any():
        raise ValueError("Duplicate team-game inputs")
    if not np.isfinite(rows[["gf", "ga", "xgf", "xga", "toi"]]).all().all():
        raise ValueError("Missing xG inputs; refresh the MoneyPuck cache")
    rows["team_abbr"] = rows.team_abbr.replace({"ARI": "UTA"})
    # MoneyPuck excludes the shootout goal. In overtime a non-tied final
    # contributes one extra goal for the winner. xG is prorated to 60 minutes
    # because team-level rows do not identify the overtime shots separately.
    extra_time = rows.toi > 3600
    rows["reg_gf"] = rows.gf - (extra_time & (rows.gf > rows.ga)).astype(int)
    rows["reg_ga"] = rows.ga - (extra_time & (rows.ga > rows.gf)).astype(int)
    for side in ("gf", "ga"):
        rows[f"reg_x{side}"] = rows[f"x{side}"] * 3600 / rows.toi.clip(lower=3600)
    rows["game_date"] = pd.to_datetime(rows.game_date)
    return rows.sort_values(["game_date", "game_id", "team_abbr"])


def _rolling(rows, by, window):
    parts = []
    for _, group in rows.groupby(by, sort=False):
        group = group.sort_values(["game_date", "game_id"]).copy()
        group[SIGNALS] = group[SIGNALS].rolling(window, min_periods=1).sum()
        group["n"] = np.minimum(np.arange(1, len(group) + 1), window)
        parts.append(group[[*by, "game_date", "n", *SIGNALS]])
    return pd.concat(parts, ignore_index=True)


def prepare(games: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = team_games(games)
    out = fixtures.copy()
    out["game_date"] = pd.to_datetime(out.game_date)
    if out.game_id.duplicated().any():
        raise ValueError("Duplicate fixtures")
    for pool, by, window in (
        ("venue", ["team_abbr", "venue"], 25),
        ("pooled", ["team_abbr"], 50),
    ):
        history = _rolling(rows, by, window)
        history = history.rename(columns={"game_date": "source_date"})
        for side in ("home", "away"):
            left = out[["game_id", "game_date", f"{side}_abbr"]].rename(
                columns={f"{side}_abbr": "team_abbr"}
            )
            left["team_abbr"] = left.team_abbr.replace({"ARI": "UTA"})
            if pool == "venue":
                left["venue"] = side
            merged = pd.merge_asof(
                left.sort_values("game_date"),
                history.sort_values("source_date"),
                left_on="game_date",
                right_on="source_date",
                by=by,
                allow_exact_matches=False,
            ).set_index("game_id")
            for column in ["n", *SIGNALS, "source_date"]:
                out[f"{side}_{pool}_{column}"] = out.game_id.map(merged[column])
    for season in sorted(out.season.unique()):
        past = rows[rows.season < season]
        if past.empty:
            raise ValueError(f"No training data before season {season}")
        mask = out.season.eq(season)
        for venue in ("home", "away", "pooled"):
            sample = past if venue == "pooled" else past[past.venue.eq(venue)]
            for column in SIGNALS:
                out.loc[mask, f"league_{venue}_{column}"] = sample[column].mean()
        prior_results = fixtures[fixtures.season < season]
        ot = prior_results[prior_results.last_period_type.isin(["OT", "SO"])]
        # Beta(50,50) keeps the small OT sample close to an even split.
        out.loc[mask, "ot_home_probability"] = (
            (ot.home_goals > ot.away_goals).sum() + 50
        ) / (len(ot) + 100)
    out["game_date"] = out.game_date.dt.date
    return out


def predict(features: pd.DataFrame, spec: Specification) -> pd.DataFrame:
    out = features[
        [
            "game_id",
            "season",
            "game_date",
            "home_abbr",
            "away_abbr",
            "home_goals",
            "away_goals",
            "last_period_type",
        ]
    ].copy()
    prefix = "reg_" if spec.overtime else ""
    strengths = {}
    for side in ("home", "away"):
        for direction in ("gf", "ga"):
            pooled_strengths = []
            for pool in ("venue", "pooled"):
                venue = side if pool == "venue" else "pooled"
                n = features[f"{side}_{pool}_n"].fillna(0)
                components = []
                for signal in (f"{prefix}{direction}", f"{prefix}x{direction}"):
                    league = features[f"league_{venue}_{signal}"]
                    count = features[f"{side}_{pool}_{signal}"].fillna(0)
                    denom = n + spec.prior_games
                    shrunk = (count + spec.prior_games * league) / denom.replace(
                        0, np.nan
                    )
                    components.append((shrunk / league).fillna(1))
                pooled_strengths.append(
                    (1 - spec.xg_weight) * components[0]
                    + spec.xg_weight * components[1]
                )
            strengths[(side, direction)] = (
                spec.venue_weight * pooled_strengths[0]
                + (1 - spec.venue_weight) * pooled_strengths[1]
            )
    for side, opponent in (("home", "away"), ("away", "home")):
        rate = (
            strengths[(side, "gf")]
            * strengths[(opponent, "ga")]
            * features[f"league_{side}_{prefix}gf"]
        )
        if spec.goalie_weight:
            rate = rate - (
                spec.goalie_weight
                * spec.xg_weight
                * features[f"{opponent}_goalie_adjustment"]
            )
        out[f"{side}_lambda"] = rate.clip(lower=0.05)
    h, a = out.home_lambda.to_numpy(), out.away_lambda.to_numpy()
    tie = skellam.pmf(0, h, a)
    ot_home = features.ot_home_probability.to_numpy() if spec.overtime else 0.5
    out["home_win_prob"] = skellam.sf(0, h, a) + tie * ot_home
    out["model_total"] = h + a + (tie if spec.overtime else 0)
    out["overtime_probability"] = tie
    out["model"] = spec.name
    return out
