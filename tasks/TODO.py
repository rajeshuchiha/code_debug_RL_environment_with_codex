"""
Task System Implementation Status

This file serves as both:
1. Developer checklist
2. Implementation guide for Codex
"""

# DONE:
# - Added tasks/easy, tasks/medium, and tasks/hard
# - Added 3 task folders per difficulty
# - Added code.py and visible_tests.json for each task
# - Added deterministic task loader in tasks/loader.py
# - inference.py includes selected task metadata and visible tests in the model prompt.
#
# PENDING:
# - Add hidden_tests.json per task for reward-only evaluation.
# - Update the environment to load hidden tests internally without exposing them
#   to inference prompts.
# - Add a task manifest or schema validation step for visible_tests.json and
#   future hidden_tests.json files.

# =========================================================
# 📁 1. Folder Setup
# =========================================================

# [ ] Create directory: tasks/
# [ ] Create subdirectories:
#       - tasks/easy/
#       - tasks/medium/
#       - tasks/hard/


# =========================================================
# 📄 2. Task Structure
# =========================================================

# Each task folder should follow:
#
# tasks/{difficulty}/task_x/
#   ├── code.py
#   ├── visible_tests.json
#
# Example:
# tasks/easy/task_1/code.py
# tasks/easy/task_1/visible_tests.json


# =========================================================
# 🟢 3. Easy Tasks
# =========================================================

# [ ] Create 2–3 tasks in tasks/easy/
# [ ] Bugs should include:
#       - syntax errors
#       - missing imports
#       - simple runtime errors
# [ ] Each task must have 2–3 visible tests


# =========================================================
# 🟡 4. Medium Tasks
# =========================================================

# [ ] Create 2–3 tasks in tasks/medium/
# [ ] Bugs should include:
#       - logic errors
#       - incorrect formulas
# [ ] Each task must have 3–5 visible tests


# =========================================================
# 🔴 5. Hard Tasks
# =========================================================

# [ ] Create 2–3 tasks in tasks/hard/
# [ ] Bugs should include:
#       - edge cases
#       - multi-step logic issues
# [ ] Each task must have 3–4 visible tests


# =========================================================
# ⚙️ 6. Task Loader Implementation
# =========================================================

# File: tasks/loader.py

# [ ] Implement function: load_task(difficulty: str)

# Expected behavior:
#   1. Read tasks/{difficulty}/
#   2. Pick a random task folder
#   3. Load:
#       - code.py (as string)
#       - visible_tests.json (as JSON)
#   4. Return dict:
#       {
#           "code": code_str,
#           "visible_tests": tests
#       }


# =========================================================
# 🧠 Suggested Implementation (Codex can follow this)
# =========================================================

def load_task(difficulty: str):
    """
    Load a random task based on difficulty.

    Args:
        difficulty (str): "easy" | "medium" | "hard"

    Returns:
        dict:
            {
                "code": str,
                "visible_tests": list
            }
    """
    import os
    import json
    import random

    base_path = os.path.join("tasks", difficulty)

    # [ ] Validate difficulty folder exists
    if not os.path.exists(base_path):
        raise ValueError(f"Invalid difficulty: {difficulty}")

    # [ ] List all task folders
    task_folders = [
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
    ]

    if not task_folders:
        raise ValueError(f"No tasks found in {difficulty}")

    # [ ] Pick random task
    selected_task = random.choice(task_folders)
    task_path = os.path.join(base_path, selected_task)

    # [ ] Read code.py
    with open(os.path.join(task_path, "code.py"), "r") as f:
        code_str = f.read()

    # [ ] Read visible_tests.json
    with open(os.path.join(task_path, "visible_tests.json"), "r") as f:
        tests = json.load(f)

    return {
        "code": code_str,
        "visible_tests": tests
    }


# =========================================================
# 🔄 7. Integration
# =========================================================

# [ ] Replace existing single example with:
#
# task = load_task(difficulty)
#
# [ ] Pass into prompt:
#       task["code"]
#       task["visible_tests"]


# =========================================================
# 🔒 8. Hidden Tests (Future)
# =========================================================

# [ ] Add file per task:
#       hidden_tests.json
#
# [ ] DO NOT expose in prompt
#
# [ ] Load only inside environment (NOT inference)
#
# [ ] Use for reward calculation only


# =========================================================
# ✅ 9. Done Criteria
# =========================================================

# [ ] Multiple tasks load correctly
# [ ] Difficulty-based selection works
# [ ] Only visible tests used in prompt
# [ ] No breaking changes to pipeline
