"""Inference script for the coding_env debugging environment."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from openai import OpenAI

try:
    from client import CodingEnv
    from models import CodingAction, CodingObservation
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from client import CodingEnv
        from models import CodingAction, CodingObservation
    except ImportError:
        from client import CodingEnv
        from models import CodingAction, CodingObservation
    
load_dotenv()

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME", "debug-python-function")
BENCHMARK = os.getenv("BENCHMARK", "coding_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
# TASK_DIFFICULTY = os.getenv("TASK_DIFFICULTY", "easy")
TASK_ID = os.getenv("TASK_ID")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous agent operating in a Python coding environment.
    Your goal is to make all visible tests pass by observing the environment
    and taking one action at a time.

    On each step you receive:
    - The current source code (with 1-based line numbers)
    - Task metadata and visible tests
    - Execution output: stdout, stderr, exceptions, stack traces
    - Inspected variable values (if requested)
    - Test results

    Available actions (return exactly one as valid JSON, no markdown):

    Run all visible tests:
      {"action_type": "run_tests"}

    Get the current stack trace:
      {"action_type": "get_stack_trace"}

    Inspect a variable by name:
      {"action_type": "inspect_variable", "var_name": "<name>"}

    Replace a single line (1-based):
      {"action_type": "edit_line", "line_no": <n>, "new_code": "<replacement>"}

    Replace a range of lines (inclusive, 1-based):
      {"action_type": "edit_block", "start_line": <n>, "end_line": <m>, "new_code": "<replacement>"}

    Insert a new line before a given line:
      {"action_type": "insert_line", "line_no": <n>, "code": "<new line>"}

    Delete a line:
      {"action_type": "delete_line", "line_no": <n>}

    Strategy:
    - Start with run_tests unless the bug is immediately obvious.
    - Use get_stack_trace or inspect_variable to gather evidence before editing.
    - Prefer edit_block for multi-line or structural fixes.
    - Prefer edit_line for isolated single-line fixes.
    - Base all decisions strictly on observed output — never assume hidden behavior.
    - Return one JSON object only. No explanation, no markdown fences.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def summarize_tests(observation: CodingObservation) -> str:
    if not observation.test_results:
        return "No tests executed yet."
    lines = []
    for index, result in enumerate(observation.test_results, start=1):
        lines.append(
            "Test {idx}: passed={passed}, input={input_val}, expected={expected}, actual={actual}".format(
                idx=index,
                passed=str(result.passed).lower(),
                input_val=result.input,
                expected=result.expected,
                actual=result.actual,
            )
        )
    return "\n".join(lines)


def summarize_task_metadata(observation: CodingObservation) -> str:
    metadata = observation.metadata or {}
    task_info = {
        "difficulty": metadata.get("difficulty"),
        "task_name": metadata.get("task_name"),
        "visible_tests": metadata.get("visible_tests", []),
    }
    return json.dumps(task_info, ensure_ascii=True, indent=2)


def format_numbered_code(code: str) -> str:
    lines = code.splitlines()
    if not lines:
        return "1 | "
    return "\n".join(f"{index} | {line}" for index, line in enumerate(lines, start=1))


def build_user_prompt(
    step: int,
    observation: CodingObservation,
    last_reward: float,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last reward: {last_reward:.2f}

        Selected task metadata and visible tests:
        {summarize_task_metadata(observation)}

        Current code with 1-based line numbers:
        {format_numbered_code(observation.code)}

        Stdout:
        {observation.stdout or "None"}

        Stderr:
        {observation.stderr or "None"}

        Exception:
        {observation.exception or "None"}

        Stack trace:
        {observation.stack_trace or "None"}

        Variables:
        {json.dumps(observation.variables, ensure_ascii=True)}

        Test results:
        {summarize_tests(observation)}

        Recent history:
        {history_block}

        Return the next best action as one JSON object.
        """
    ).strip()


def parse_action_response(raw_text: str) -> Dict[str, Any]:
    candidate = raw_text.strip()
    if "```" in candidate:
        parts = [part.strip() for part in candidate.split("```") if part.strip()]
        candidate = parts[0]
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
    return json.loads(candidate)


def fallback_action(step: int, observation: CodingObservation) -> Dict[str, Any]:
    
    return {"action_type": "run_tests"}


def get_model_action(
    client: OpenAI,
    step: int,
    observation: CodingObservation,
    last_reward: float,
    history: List[str],
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(step, observation, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        return parse_action_response(content)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return fallback_action(step, observation)


def format_action(action_payload: Dict[str, Any]) -> str:
    return json.dumps(action_payload, ensure_ascii=True, separators=(",", ":"))


def compute_score(observation: CodingObservation) -> float:
    if not observation.test_results:
        return 0.01   # not 0.0
    passed = sum(1 for result in observation.test_results if result.passed)
    total = len(observation.test_results)
    if total == 0:
        return 0.01
    raw = passed / total
    # clamp strictly inside (0, 1)
    return max(0.01, min(0.99, raw))


async def create_env() -> CodingEnv:
    if IMAGE_NAME:
        return await CodingEnv.from_docker_image(image=IMAGE_NAME)
    env = CodingEnv(base_url=os.getenv("ENV_BASE_URL", "http://localhost:8000"))
    await env.connect()
    return env


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    env = await create_env()
    
    all_rewards: List[float] = []
    all_scores: List[float] = []

    try:
        
        for difficulty in ["easy", "medium", "hard"]:
            
            history: List[str] = []
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False
            observation = CodingObservation()

            reset_kwargs: Dict[str, Any] = {"difficulty": difficulty}  # ← uses loop var
            if TASK_ID:
                reset_kwargs["task_name"] = TASK_ID

            result = await env.reset(**reset_kwargs)
            observation = result.observation
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action_payload = get_model_action(client, step, observation, last_reward, history)
                error: Optional[str] = None

                try:
                    action = CodingAction(**action_payload)
                except Exception as exc:
                    error = str(exc)
                    action_payload = fallback_action(step, observation)
                    action = CodingAction(**action_payload)

                result = await env.step(action)
                observation = result.observation

                reward = float(result.reward or 0.0)
                done = result.done
                error = error or (observation.exception if observation.exception else None)

                rewards.append(reward)
                steps_taken = step
                last_reward = reward

                action_str = format_action(action_payload)
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                history.append(
                    f"Step {step}: {action_str} -> reward {reward:+.2f}, tests={summarize_tests(observation)}"
                )

                if done:
                    break

            score = compute_score(observation)
            success = all(result.passed for result in observation.test_results)

            all_rewards.extend(rewards)
            all_scores.append(score)

            # ✅ Log per-difficulty result
            print(f"[DIFFICULTY] level={difficulty} score={score:.3f} success={str(success).lower()}", flush=True)
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
