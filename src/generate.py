"""
src/generate.py

1. Define the behaviour we want the final prompt to achieve.
2. Generate user queries and expected answers for that behaviour.
3. Save the generated cases as evaluation data.
4. The evaluation data will later measure prompt performance.
5. The evaluation data should not depend on the current candidate prompt.
"""

import json
import os

import dspy
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from constants import (
    EVAL_DATA_PATH,
    GENERATED_CASE_COUNT,
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


# 3. Define one generated dataset row
class GeneratedCase(BaseModel):
    input: str = Field(
        description="Realistic synthetic user input"
    )

    output: str | None = Field(
        description="Exact expected output, or null if no valid answer exists"
    )


# 4. Define the dataset-generation task
class GenerateCases(dspy.Signature):
    """Generate diverse and unambiguous evaluation cases."""

    task_objective: str = dspy.InputField(
        desc="The behaviour that the evaluated prompt should achieve"
    )

    case_requirements: str = dspy.InputField(
        desc="Requirements for the evaluation cases to generate"
    )

    count: int = dspy.InputField(
        desc="Number of evaluation cases to generate"
    )

    cases: list[GeneratedCase] = dspy.OutputField(
        desc="Generated evaluation cases"
    )


# 5. Create the dataset generator
generate_cases = dspy.Predict(GenerateCases)


# 6. Load the task independently of the candidate prompt
with INPUTS_PATH.open("r", encoding="utf-8") as file:
    inputs = yaml.safe_load(file)


# 7. Generate evaluation cases
prediction = generate_cases(
    task_objective=inputs["task_objective"],
    case_requirements=inputs["case_requirements"],
    count=GENERATED_CASE_COUNT,
)


# 8. Convert Pydantic objects into regular dictionaries
cases = [
    case.model_dump()
    for case in prediction.cases
]


# 9. Save as JSON
# "w" means overwrite mode
with EVAL_DATA_PATH.open("w", encoding="utf-8") as file:
    json.dump(
        cases,
        file,
        indent=2,
        ensure_ascii=False,
    )
