import json
import os

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# 1. Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY was not found in the .env file")


# 2. Configure DSPy
lm = dspy.LM(
    "openai/gpt-5-nano",
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


# 4. Define the DSPy task
class GenerateCases(dspy.Signature):
    """Generate diverse and unambiguous evaluation cases for the given task."""

    fixed_prompt: str = dspy.InputField(
        desc="The permanent system or task instructions"
    )

    variable_prompt: str = dspy.InputField(
        desc="An example request or description of cases to generate"
    )

    count: int = dspy.InputField(
        desc="Number of evaluation cases to generate"
    )

    cases: list[GeneratedCase] = dspy.OutputField(
        desc="Generated evaluation cases"
    )


# 5. Create the predictor
generate_cases = dspy.Predict(GenerateCases)


# 6. Generate cases
prediction = generate_cases(
    fixed_prompt=(
        "Answer factual geography questions accurately and concisely. "
        "Return only the answer."
    ),
    variable_prompt=(
        "Generate different ways users might ask for the capital of France. "
        "Every case must have Paris as the expected answer."
    ),
    count=10,
)


# 7. Convert Pydantic objects into regular dictionaries
cases = [case.model_dump() for case in prediction.cases]


# 8. Save as JSON 
# "w" -> overwrite mode
with open("src/train_data.json", "w", encoding="utf-8") as file:
    json.dump(cases, file, indent=2, ensure_ascii=False)