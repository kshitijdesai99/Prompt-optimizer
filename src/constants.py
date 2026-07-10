"""Shared project configuration."""

from pathlib import Path

SRC_DIR = Path(__file__).parent

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
GENERATION_MODEL_NAME = "openai/gpt-5.6-luna"
MODEL_NAME = "openai/gpt-5-nano"

INPUTS_PATH = SRC_DIR / "inputs.yaml"
EVAL_DATA_PATH = SRC_DIR / "eval_data.json"
EVAL_RESULTS_PATH = SRC_DIR / "eval_results.json"
OPTIMIZATION_RESULTS_ROOT = SRC_DIR / "results"
OPTIMIZATION_RESULTS_FILENAME = "optimization_results.json"

GENERATED_CASE_COUNT = 30
MIN_OPTIMIZATION_CASES = 5
TRAINING_SPLIT = 0.8
RANDOM_SEED = 42

DEFAULT_OPTIMIZER = "mipro_v2"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

MIPRO_AUTO_MODE = "light"
MIPRO_MAX_BOOTSTRAPPED_DEMOS = 0
MIPRO_MAX_LABELED_DEMOS = 0
MIPRO_VERBOSE = True

GEPA_MAX_METRIC_CALLS = 50
