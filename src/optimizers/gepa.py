"""GEPA optimizer configuration."""

import dspy

from constants import GEPA_MAX_METRIC_CALLS


def create_optimizer(metric, lm: dspy.LM):
    """Create a lightweight reflective prompt optimizer."""
    def gepa_metric(
        gold,
        prediction,
        trace=None,
        prediction_name=None,
        prediction_trace=None,
    ):
        return metric(gold, prediction, trace)

    return dspy.GEPA(
        metric=gepa_metric,
        reflection_lm=lm,
        max_metric_calls=GEPA_MAX_METRIC_CALLS,
    )
