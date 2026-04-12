---
title: Coding Env Debugging Environment Server
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - debugging
  - reinforcement-learning
---

# Coding Env Debugging Environment

`coding_env` is an OpenEnv environment for deterministic code-debugging reinforcement learning. The agent receives buggy Python source code, edits it line by line, runs tests, inspects variables, and uses stack traces to repair the target function.

Tasks are loaded from `tasks/{difficulty}/task_x/` through `tasks.loader.load_task(...)`. Each task provides a `code.py` file and a `visible_tests.json` file. The episode ends when all visible tests pass or the maximum step count is reached.

## Action Space

Actions are typed through `CodingAction` in `models.py`.

```python
CodingAction(action_type="edit_line", line_no=<n>, new_code="<replacement>")
CodingAction(action_type="edit_block", start_line=<n>, end_line=<m>, new_code="<replacement>")
CodingAction(action_type="insert_line", line_no=<n>, code="<new line>")
CodingAction(action_type="delete_line", line_no=<n>)
CodingAction(action_type="run_tests")
CodingAction(action_type="inspect_variable", var_name="total")
CodingAction(action_type="get_stack_trace")
```

Supported action types:

- `edit_line(line_no, new_code)`: Replace a 1-based line with new code.
- `edit_block(start_line, end_line, new_code)`: Replace/Insert new code between start_line and end_line (1-based inclusive).
- `insert_line(line_no, code)`: Insert code before a 1-based line.
- `delete_line(line_no)`: Delete a 1-based line.
- `run_tests()`: Execute the deterministic test suite.
- `inspect_variable(var_name)`: Capture a variable value during execution.
- `get_stack_trace()`: Return the last stack trace, or run tests if none exists.

## Observation Space

Each `CodingObservation` includes:

- `code`: Current source code.
- `stdout`: Captured stdout from the last subprocess execution.
- `stderr`: Captured stderr from the last subprocess execution.
- `exception`: Last exception message, if any.
- `stack_trace`: Last traceback, if any.
- `test_results`: Per-test dictionaries containing `input`, `expected`, `actual`, and `passed`.
- `variables`: Values captured by variable inspection.
- `step_count`: Current episode step count.
- `done`: Whether the episode is complete.
- `reward`: Reward from the most recent action.
- `metadata`: Includes `max_steps`, `tests_passed`, `total_tests`, `difficulty`, `task_name`, and `visible_tests`.

## Task System

Tasks live under `tasks/` and are grouped by difficulty:

```text
tasks/
|-- easy/
|   |-- task_1/
|   |   |-- code.py
|   |   `-- visible_tests.json
|-- medium/
`-- hard/
```

Each `visible_tests.json` entry uses this shape:

```json
{
  "function_name": "final_price",
  "input": [100, 25],
  "expected": 75.0
}
```

`function_name` tells the runner which function to call. `input` is unpacked as positional arguments. `expected` is compared to the actual return value.

You can select a task during reset:

```python
result = await env.reset(difficulty="medium", task_name="task_1")
```

If `task_name` is omitted, the loader uses deterministic seeded selection by difficulty.

## Reward Function

The environment uses the project reward policy:

- `+1.0` when all tests pass.
- `+0.3` when the number of passed tests improves.
- `-0.2` for runtime or syntax errors.
- `-0.05` per step.
- `-0.5` for timeouts.
- `-0.1` for invalid code edits.
- `-0.1*change_ratio` for limiting large code edits.

Code execution always happens in a subprocess with a 2-second timeout.

## Quick Start

Start the server:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Use the async OpenEnv client:

```python
import asyncio

from coding_env.client import CodingEnv
from coding_env.models import CodingAction


async def main():
    env = CodingEnv(base_url="http://localhost:8000")
    await env.connect()
    try:
        result = await env.reset(difficulty="easy", task_name="task_1")
        print(result.observation.code)
        print(result.observation.metadata["visible_tests"])

        result = await env.step(CodingAction(action_type="run_tests"))
        print(result.observation.exception)
        print(result.observation.stack_trace)

        result = await env.step(
            CodingAction(
                action_type="edit_line",
                line_no=1,
                new_code="def add_numbers(a, b):",
            )
        )
        result = await env.step(
            CodingAction(action_type="edit_line", line_no=2, new_code="    return a + b")
        )
        result = await env.step(CodingAction(action_type="run_tests"))
        print(result.done, result.reward)
        print([test.passed for test in result.observation.test_results])
    finally:
        await env.close()


asyncio.run(main())
```

## Inference Script

`inference.py` runs an async model-driven debugging loop. It expects the model to return exactly one JSON action per step.

Required or useful environment variables:

```env
HF_TOKEN=<your-token>
API_KEY=<optional-api-key>
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV_BASE_URL=http://localhost:8000
LOCAL_IMAGE_NAME=coding_env-env:latest
MAX_STEPS=20
TASK_DIFFICULTY=easy
TASK_ID=task_1
OPEN_API_KEY=<your-token>
```

Run inference against a local server:

```bash
python inference.py
```

If `LOCAL_IMAGE_NAME` or `IMAGE_NAME` is set, `inference.py` starts the environment from Docker through `CodingEnv.from_docker_image(...)`. Otherwise it connects to `ENV_BASE_URL`.

`inference.py` passes `TASK_DIFFICULTY` and optional `TASK_ID` to `env.reset(...)`. It also includes the selected task metadata and `visible_tests` in the model prompt so the model can reason from the allowed public tests before choosing actions.

## Docker

Build the Docker image from the environment directory:

```bash
docker build -t coding_env-env:latest -f server/Dockerfile .
```

Run it:

```bash
docker run --rm -p 8000:8000 coding_env-env:latest
```

Then connect with:

```bash
ENV_BASE_URL=http://localhost:8000 python inference.py
```

## OpenEnv Deployment

Validate and push with the OpenEnv CLI:

```bash
openenv validate
openenv push
```

The deployed space exposes:

- `/web`: Interactive OpenEnv web UI.
- `/docs`: FastAPI/OpenAPI documentation.
- `/health`: Health check.
- `/ws`: WebSocket endpoint used by the OpenEnv client.

## Project Structure

```text
coding_env/
|-- __init__.py
|-- README.md
|-- client.py
|-- inference.py
|-- models.py
|-- openenv.yaml
|-- pyproject.toml
|-- tasks/
|   |-- __init__.py
|   |-- loader.py
|   |-- easy/
|   |-- medium/
|   `-- hard/
|-- uv.lock
`-- server/
    |-- __init__.py
    |-- app.py
    |-- coding_env_environment.py
    |-- Dockerfile
    `-- requirements.txt
```
