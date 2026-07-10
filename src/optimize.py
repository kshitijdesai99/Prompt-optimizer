"""
src/optimize.py

1. Load the generated dataset.
2. Split it into training and test cases.
3. Optimize only the variable prompt using the training cases.
4. Evaluate the optimized program on unseen test cases.
5. Save the optimization results.
"""

import json
import os
import random
from datetime import datetime

import dspy
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from constants import (
    DEFAULT_OPTIMIZER,
    EVAL_DATA_PATH,
    INPUTS_PATH,
    MIN_OPTIMIZATION_CASES,
    MODEL_NAME,
    OPENAI_API_KEY_ENV,
    OPTIMIZATION_RESULTS_FILENAME,
    OPTIMIZATION_RESULTS_ROOT,
    RANDOM_SEED,
    TIMESTAMP_FORMAT,
    TRAINING_SPLIT,
)
from optimizers.gepa import create_optimizer as create_gepa
from optimizers.mipro_v2 import create_optimizer as create_mipro_v2


# 1. Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv(OPENAI_API_KEY_ENV)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY was not found in the .env file")


# 2. Configure DSPy
lm = dspy.LM(
    MODEL_NAME,
    api_key=OPENAI_API_KEY,
)

dspy.configure(lm=lm)


# Defaults to MIPROv2. Set OPTIMIZER_NAME to "gepa" when running this
# script to select the reflective optimizer.
OPTIMIZER_NAME = os.getenv(
    "OPTIMIZER_NAME",
    DEFAULT_OPTIMIZER,
)

RUN_TIMESTAMP = datetime.now().strftime(TIMESTAMP_FORMAT)
RESULTS_DIR = OPTIMIZATION_RESULTS_ROOT / f"{RUN_TIMESTAMP}_{OPTIMIZER_NAME}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OPTIMIZATION_RESULTS_PATH = RESULTS_DIR / OPTIMIZATION_RESULTS_FILENAME

OPTIMIZER_FACTORIES = {
    "mipro_v2": create_mipro_v2,
    "gepa": create_gepa,
}

if OPTIMIZER_NAME not in OPTIMIZER_FACTORIES:
    available_optimizers = ", ".join(OPTIMIZER_FACTORIES)
    raise ValueError(
        f"Unknown optimizer: {OPTIMIZER_NAME}. "
        f"Choose one of: {available_optimizers}."
    )


# 4. Define one dataset row
class EvaluationCase(BaseModel):
    input: str = Field(
        description="User input given to the prompt"
    )

    output: str | None = Field(
        description="Expected output"
    )


# 5. Load the fixed and initial variable prompts
with INPUTS_PATH.open("r", encoding="utf-8") as file:
    inputs = yaml.safe_load(file)

FIXED_PROMPT = inputs["fixed_prompt"]
INITIAL_VARIABLE_PROMPT = inputs["variable_prompt"]


# 6. Define the DSPy task
class AnswerQuestion(dspy.Signature):
    fixed_prompt: str = dspy.InputField(
        desc="Permanent instructions that must always be followed"
    )

    input: str = dspy.InputField(
        desc="The user's request"
    )

    output: str | None = dspy.OutputField(
        desc="The final answer"
    )


# 7. Define the program that will be optimized
class PromptProgram(dspy.Module):
    def __init__(self, variable_prompt: str):
        super().__init__()

        # The variable prompt becomes the optimizable
        # instruction of this DSPy predictor.
        signature = AnswerQuestion.with_instructions(
            variable_prompt
        )

        self.answer_question = dspy.Predict(signature)

    def forward(self, input: str):
        return self.answer_question(
            fixed_prompt=FIXED_PROMPT,
            input=input,
        )


# 8. Normalize outputs before comparing them
def normalize_output(value: str | None) -> str | None:
    """
    Ignore capitalization and surrounding whitespace.

    Examples:
        "Paris"   -> "paris"
        " PARIS " -> "paris"

    Quotation marks are intentionally not removed because
    the expected output must contain only the answer.
    """

    if value is None:
        return None

    normalized = value.strip().casefold()

    if not normalized:
        return None

    return normalized


# 9. Define the optimization metric
def exact_match(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> bool:
    expected_output = normalize_output(example.output)
    predicted_output = normalize_output(prediction.output)

    return predicted_output == expected_output


# 10. Load the generated dataset
if not EVAL_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset was not found at: {EVAL_DATA_PATH}\n"
        "Run src/generate.py first."
    )

with EVAL_DATA_PATH.open("r", encoding="utf-8") as file:
    raw_cases = json.load(file)


# 11. Validate the dataset
validated_cases = [
    EvaluationCase.model_validate(case)
    for case in raw_cases
]

if len(validated_cases) < MIN_OPTIMIZATION_CASES:
    raise ValueError(
        f"At least {MIN_OPTIMIZATION_CASES} cases are required."
    )


# 12. Convert cases into DSPy examples
examples = [
    dspy.Example(
        input=case.input,
        output=case.output,
    ).with_inputs("input")
    for case in validated_cases
]


# 13. Shuffle the cases predictably
# Using a fixed seed means we get the same split every time.
random_generator = random.Random(RANDOM_SEED)
random_generator.shuffle(examples)


# 14. Split cases into training and test sets
split_index = int(len(examples) * TRAINING_SPLIT)

trainset = examples[:split_index]
testset = examples[split_index:]

print(f"Training cases: {len(trainset)}")
print(f"Test cases:     {len(testset)}")


# 15. Create the initial program
initial_program = PromptProgram(
    variable_prompt=INITIAL_VARIABLE_PROMPT
)


# 16. Create an evaluator for the optimized program's test cases
test_evaluator = dspy.Evaluate(
    devset=testset,
    metric=exact_match,
    display_progress=True,
)


# 17. Create the selected prompt optimizer
optimizer = OPTIMIZER_FACTORIES[OPTIMIZER_NAME](
    metric=exact_match,
    lm=lm,
)


# 18. Optimize the variable prompt
print()
print("Optimizing variable prompt...")

optimized_program = optimizer.compile(
    initial_program,
    trainset=trainset,
)


# 19. Evaluate the optimized program
print()
print("Evaluating optimized prompt...")

optimized_evaluation = test_evaluator(
    optimized_program
)


# 20. Read the optimized variable prompt
optimized_variable_prompt = (
    optimized_program
    .answer_question
    .signature
    .instructions
)


# 21. Convert evaluation results into dictionaries
def serialize_results(evaluation) -> list[dict]:
    results = []

    for example, prediction, score in evaluation.results:
        results.append(
            {
                "input": example.input,
                "expected_output": example.output,
                "predicted_output": getattr(
                    prediction,
                    "output",
                    None,
                ),
                "correct": bool(score),
            }
        )

    return results


optimized_results = serialize_results(
    optimized_evaluation
)


# 22. Build the optimization report
report = {
    "model": MODEL_NAME,
    "optimizer": OPTIMIZER_NAME,
    "fixed_prompt": FIXED_PROMPT,
    "original_prompt": INITIAL_VARIABLE_PROMPT,
    "optimized_variable_prompt": optimized_variable_prompt,
    "training_cases": len(trainset),
    "test_cases": len(testset),
    "optimized_accuracy_percentage": float(
        optimized_evaluation.score
    ),
    "optimized_results": optimized_results,
}


# 23. Save the optimization report
with OPTIMIZATION_RESULTS_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        report,
        file,
        indent=2,
        ensure_ascii=False,
    )
