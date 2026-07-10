# Prompt Optimizer

A small learning project for improving a prompt with DSPy.

You provide a task and a starting prompt. The project generates test cases,
measures the starting prompt, and produces an optimized prompt you can copy
into another application.

## Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Install dependencies:

```bash
uv sync
```

## Workflow

Run the scripts in this order:

```bash
uv run python src/generate.py
uv run python src/evaluate.py
uv run python src/optimize.py
```

1. `generate.py` creates evaluation cases from your task description.
2. `evaluate.py` measures how well your starting prompt performs.
3. `optimize.py` improves the variable prompt and saves a result report.

## Edit your prompts

Edit [src/inputs.yaml](src/inputs.yaml):

```yaml
task_objective: >
  Describe the behaviour you want the final prompt to achieve.

case_requirements: >
  Describe the kinds of evaluation cases to generate.

fixed_prompt: >
  Instructions that must never change.

variable_prompt: >
  The starting prompt that the optimizer may improve.
```

Use Markdown freely inside YAML block values if it makes your prompts easier to
read.

## Choose an optimizer

`MIPROv2` is the default. It is the best fit when you want a reusable final
prompt as text.

```bash
uv run python src/optimize.py
```

Use GEPA when you want a more reflective, more expensive prompt search:

```bash
OPTIMIZER_NAME=gepa uv run python src/optimize.py
```

## Find the final prompt

Each optimization run creates a timestamped folder under `src/results/`, for
example:

```text
src/results/20260711_120000_mipro_v2/
└── optimization_results.json
```

Open `optimization_results.json` and copy the value of
`optimized_variable_prompt`. The file also includes the original prompt,
accuracy, and individual test results.

To use the result outside this project, combine `fixed_prompt` and
`optimized_variable_prompt` in your target application.

## Project structure

```text
src/
├── constants.py          # Shared settings
├── inputs.yaml           # Your task and starting prompts
├── generate.py           # Creates evaluation cases
├── evaluate.py           # Measures the starting prompt
├── optimize.py           # Produces an optimized prompt
└── optimizers/           # MIPROv2 and GEPA configurations
```
