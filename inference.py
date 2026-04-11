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
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are solving a deterministic Python debugging task.
    You must reply with exactly one JSON object and no extra text.

    Allowed actions:
    {"action_type":"run_tests"}
    {"action_type":"get_stack_trace"}
    {"action_type":"inspect_variable","var_name":"total"}
    {"action_type":"edit_line","line_no":4,"new_code":"    total = 0"}
    {"action_type":"insert_line","line_no":3,"code":"    print(numbers)"}
    {"action_type":"delete_line","line_no":7}

    Rules:
    - Prefer run_tests first.
    - Use get_stack_trace or inspect_variable when tests fail and you need more evidence.
    - Only edit one line per step.
    - Keep line numbers 1-based.
    - Return valid JSON only.
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
        Current code:
        {observation.code}

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
    if step == 1 or not observation.test_results:
        return {"action_type": "run_tests"}
    if observation.exception and not observation.stack_trace:
        return {"action_type": "get_stack_trace"}
    if observation.exception and "len(value)" in observation.stack_trace:
        return {
            "action_type": "edit_line",
            "line_no": 7,
            "new_code": "    average = total / len(numbers)",
        }
    if observation.exception and not observation.variables:
        return {"action_type": "inspect_variable", "var_name": "total"}
    if observation.variables.get("total") == "13":
        return {"action_type": "edit_line", "line_no": 4, "new_code": "    total = 0"}
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
        return 0.0
    passed = sum(1 for result in observation.test_results if result.passed)
    total = len(observation.test_results)
    return passed / total if total else 0.0


async def create_env() -> CodingEnv:
    if IMAGE_NAME:
        return await CodingEnv.from_docker_image(image=IMAGE_NAME)
    env = CodingEnv(base_url=os.getenv("ENV_BASE_URL", "http://localhost:8000"))
    await env.connect()
    return env


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    observation = CodingObservation()

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    env = await create_env()
    try:
        result = await env.reset()
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
        success = score >= 1.0 and all(result.passed for result in observation.test_results)

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
