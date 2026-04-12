"""Deterministic task loader for coding_env debugging tasks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional


VALID_DIFFICULTIES = {"easy", "medium", "hard"}
TASKS_ROOT = Path(__file__).resolve().parent


def load_task(
    difficulty: str,
    task_name: Optional[str] = None,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """Load a debugging task by difficulty.

    Args:
        difficulty: One of "easy", "medium", or "hard".
        task_name: Optional exact task folder name. When omitted, a deterministic
            seeded choice is made from the available folders.
        seed: Seed for deterministic task selection. Use None for non-deterministic
            random selection.

    Returns:
        A dict containing code, visible tests, difficulty, and task name.
    """
    normalized_difficulty = difficulty.lower().strip()
    if normalized_difficulty not in VALID_DIFFICULTIES:
        raise ValueError(
            f"Invalid difficulty: {difficulty}. Expected one of: {sorted(VALID_DIFFICULTIES)}"
        )

    difficulty_path = TASKS_ROOT / normalized_difficulty
    if not difficulty_path.is_dir():
        raise ValueError(f"Difficulty folder does not exist: {difficulty_path}")

    task_folders = sorted(path for path in difficulty_path.iterdir() if path.is_dir())
    if not task_folders:
        raise ValueError(f"No tasks found in {normalized_difficulty}")

    if task_name is not None:
        selected_task = difficulty_path / task_name
        if not selected_task.is_dir():
            raise ValueError(f"Task not found: {normalized_difficulty}/{task_name}")
    else:
        rng = random.Random(seed) if seed is not None else random
        selected_task = rng.choice(task_folders)

    code_path = selected_task / "code.py"
    tests_path = selected_task / "visible_tests.json"

    if not code_path.is_file():
        raise FileNotFoundError(f"Missing task code file: {code_path}")
    if not tests_path.is_file():
        raise FileNotFoundError(f"Missing visible tests file: {tests_path}")

    with code_path.open("r", encoding="utf-8") as code_file:
        code = code_file.read() 

    with tests_path.open("r", encoding="utf-8") as tests_file:
        visible_tests = json.load(tests_file)

    return {
        "difficulty": normalized_difficulty,
        "task_name": selected_task.name,
        "code": code,
        "visible_tests": visible_tests,
    }

