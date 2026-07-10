"""
src/evaluate.py

1. Load evaluation cases from eval_data.json.
2. Run the fixed prompt and variable prompt on every user input.
3. Compare each predicted output with the expected output.
4. Calculate the current prompt's accuracy.
5. Save detailed evaluation results.
"""

import json
import os

import dspy
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from constants import (
    EVAL_DATA_PATH,
    EVAL_RESULTS_PATH,
    INPUTS_PATH,
    MODEL_NAME,
    OPENAI_API_KEY_ENV,
)


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


# 4. Define one evaluation dataset row
class EvaluationCase(BaseModel):
    input: str = Field(
        description="User input given to the prompt"
    )

    output: str | None = Field(
        description="Expected output"
    )


# 5. Load the fixed and variable prompts
with INPUTS_PATH.open("r", encoding="utf-8") as file:
    inputs = yaml.safe_load(file)

FIXED_PROMPT = inputs["fixed_prompt"]
VARIABLE_PROMPT = inputs["variable_prompt"]


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


# 7. Define the program being evaluated
class PromptProgram(dspy.Module):
    def __init__(self, variable_prompt: str):
        super().__init__()

        # The variable prompt becomes the signature's instructions.
        #
        # Later, a DSPy optimizer can optimize these instructions.
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
    Remove irrelevant formatting differences.

    Examples:
        "Paris"   -> "paris"
        " PARIS " -> "paris"
        None      -> None
    """

    if value is None:
        return None

    normalized = value.strip().casefold()

    if not normalized:
        return None

    return normalized


# 9. Define the evaluation metric
def exact_match(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> bool:
    expected_output = normalize_output(example.output)
    predicted_output = normalize_output(prediction.output)

    return predicted_output == expected_output


# 10. Load evaluation data
if not EVAL_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Evaluation data was not found at: {EVAL_DATA_PATH}\n"
        "Run src/generate.py first."
    )

with EVAL_DATA_PATH.open("r", encoding="utf-8") as file:
    raw_cases = json.load(file)


# 11. Validate the loaded cases
validated_cases = [
    EvaluationCase.model_validate(case)
    for case in raw_cases
]

if not validated_cases:
    raise ValueError("No evaluation cases were found in eval_data.json")


# 12. Convert cases into DSPy examples
evalset = [
    dspy.Example(
        input=case.input,
        output=case.output,
    ).with_inputs("input")
    for case in validated_cases
]


# 13. Create the current prompt program
program = PromptProgram(
    variable_prompt=VARIABLE_PROMPT
)


# 14. Create the evaluator
evaluator = dspy.Evaluate(
    devset=evalset,
    metric=exact_match,
    display_progress=True,
)


# 15. Evaluate the current prompt
evaluation = evaluator(program)


# 16. Convert detailed results into dictionaries
results = []

for example, prediction, score in evaluation.results:
    predicted_output = getattr(
        prediction,
        "output",
        None,
    )

    result = {
        "input": example.input,
        "expected_output": example.output,
        "predicted_output": predicted_output,
        "correct": bool(score),
    }

    results.append(result)


# 17. Build the evaluation report
correct_count = sum(
    result["correct"]
    for result in results
)

total_count = len(results)

report = {
    "model": MODEL_NAME,
    "fixed_prompt": FIXED_PROMPT,
    "variable_prompt": VARIABLE_PROMPT,
    "total_cases": total_count,
    "correct_cases": correct_count,
    "incorrect_cases": total_count - correct_count,
    "accuracy_percentage": float(evaluation.score),
    "results": results,
}


# 18. Save the evaluation report
with EVAL_RESULTS_PATH.open("w", encoding="utf-8") as file:
    json.dump(
        report,
        file,
        indent=2,
        ensure_ascii=False,
    )
