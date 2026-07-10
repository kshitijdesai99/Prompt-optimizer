"""MIPROv2 optimizer configuration."""

import dspy

from constants import (
    MIPRO_AUTO_MODE,
    MIPRO_MAX_BOOTSTRAPPED_DEMOS,
    MIPRO_MAX_LABELED_DEMOS,
    MIPRO_VERBOSE,
)


def create_optimizer(metric, lm: dspy.LM):
    """Create an instruction-only MIPROv2 optimizer."""
    return dspy.MIPROv2(
        metric=metric,
        auto=MIPRO_AUTO_MODE,
        prompt_model=lm,
        task_model=lm,
        max_bootstrapped_demos=MIPRO_MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=MIPRO_MAX_LABELED_DEMOS,
        verbose=MIPRO_VERBOSE,
    )
