"""Test rank-selection logic used in adaptive PowerSGD training.

Focus on pure decision logic only:
- calibration best-rank selection
- calibration-best to effective max-rank mapping
- ACCORDION epoch-end target-rank decision
"""

from typing import Any, Dict, List

from src.train.train import (
    apply_calibration_best_as_max_rank,
    decide_accordion_target_rank,
    select_best_rank_from_calibration_stats,
)


CalibrationStat = Dict[str, Any]


def test_select_best_rank_prefers_max_feasible() -> None:
    """Verify that calibration selects the maximum feasible rank."""
    stats: List[CalibrationStat] = [
        {"rank": 1, "comm_ratio": 0.40},
        {"rank": 2, "comm_ratio": 0.29},
        {"rank": 4, "comm_ratio": 0.35},
        {"rank": 8, "comm_ratio": 0.31},
    ]
    best = select_best_rank_from_calibration_stats(stats=stats, ranks=[1, 2, 4, 8], threshold=0.30)
    assert best == 2


def test_select_best_rank_bumps_one_to_two_when_possible() -> None:
    """Verify that selection bumps rank 1 to rank 2 when available."""
    stats: List[CalibrationStat] = [
        {"rank": 1, "comm_ratio": 0.20},
        {"rank": 2, "comm_ratio": 0.35},
        {"rank": 4, "comm_ratio": 0.50},
    ]
    best = select_best_rank_from_calibration_stats(stats=stats, ranks=[1, 2, 4], threshold=0.30)
    assert best == 2


def test_apply_calibration_best_as_max_rank_clamps_and_preserves_order() -> None:
    """Verify that calibration best-rank is clamped and mapped to effective max-rank."""
    min_rank, max_rank = apply_calibration_best_as_max_rank(min_rank=1, max_rank=8, best_rank=2)
    assert (min_rank, max_rank) == (1, 2)

    min_rank, max_rank = apply_calibration_best_as_max_rank(min_rank=2, max_rank=8, best_rank=1)
    assert (min_rank, max_rank) == (2, 2)


def test_decide_accordion_force_max_before_epoch_20() -> None:
    """Verify that force-max window always selects max-rank."""
    target, change_rate, regime = decide_accordion_target_rank(
        epoch=19,
        current_rank=1,
        min_rank=1,
        max_rank=8,
        end_norm=10.0,
        past_value=9.0,
        lr_cooldown=0,
        rank_cooldown=0,
    )
    assert target == 8
    assert change_rate is None
    assert regime == "force_max"


def test_decide_accordion_rank_cooldown_holds_rank() -> None:
    """Verify that rank cooldown keeps the current rank."""
    target, change_rate, regime = decide_accordion_target_rank(
        epoch=30,
        current_rank=4,
        min_rank=1,
        max_rank=8,
        end_norm=10.0,
        past_value=9.0,
        lr_cooldown=0,
        rank_cooldown=3,
    )
    assert target == 4
    assert change_rate is None
    assert regime == "rank_cooldown"


def test_decide_accordion_critical_vs_not_critical() -> None:
    """Verify critical and non-critical branches at the change-rate threshold."""
    target, change_rate, regime = decide_accordion_target_rank(
        epoch=30,
        current_rank=4,
        min_rank=1,
        max_rank=8,
        end_norm=20.0,
        past_value=10.0,
        lr_cooldown=0,
        rank_cooldown=0,
    )
    assert target == 8
    assert abs(change_rate - 1.0) < 1e-12
    assert regime == "critical"

    target, change_rate, regime = decide_accordion_target_rank(
        epoch=30,
        current_rank=8,
        min_rank=1,
        max_rank=8,
        end_norm=11.0,
        past_value=10.0,
        lr_cooldown=0,
        rank_cooldown=0,
    )
    assert target == 1
    assert abs(change_rate - 0.1) < 1e-12
    assert regime == "not_critical"


def test_decide_accordion_insufficient_history_keeps_rank() -> None:
    """Verify that missing history keeps the current rank."""
    target, change_rate, regime = decide_accordion_target_rank(
        epoch=30,
        current_rank=2,
        min_rank=1,
        max_rank=8,
        end_norm=10.0,
        past_value=None,
        lr_cooldown=0,
        rank_cooldown=0,
    )
    assert target == 2
    assert change_rate == 0.0
    assert regime == "insufficient_history"
