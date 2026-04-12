# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Production-grade deterministic code debugging environment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import traceback
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import CodingAction, CodingObservation, TestCaseResult
    from tasks import load_task
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models import CodingAction, CodingObservation, TestCaseResult
    from tasks import load_task


EXECUTION_TIMEOUT_S = 2.0
TEMP_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".runtime")


class CodingEnvironment(Environment):
    """Deterministic RL environment for debugging Python code."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        max_steps: int = 20,
        difficulty: str = "easy",
        task_name: Optional[str] = None,
        task_seed: Optional[int] = 0,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.task_name = task_name
        self.task_seed = task_seed
        os.makedirs(TEMP_ROOT, exist_ok=True)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._code_lines: List[str] = []
        self._task: Dict[str, Any] = {}
        self._test_cases: List[Dict[str, Any]] = []
        self._last_stdout = ""
        self._last_stderr = ""
        self._last_exception = ""
        self._last_stack_trace = ""
        self._last_test_results: List[TestCaseResult] = []
        self._last_variables: Dict[str, Any] = {}
        self._passed_tests = 0
        self._all_tests_passed = False
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CodingObservation:
        difficulty = str(kwargs.pop("difficulty", self.difficulty))
        task_name = cast(Optional[str], kwargs.pop("task_name", self.task_name))
        task_seed = self.task_seed if seed is None else seed
        self._task = load_task(difficulty, task_name=task_name, seed=task_seed)
        self._test_cases = cast(List[Dict[str, Any]], self._task["visible_tests"])
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._code_lines = cast(str, self._task["code"]).splitlines()
        self._last_stdout = ""
        self._last_stderr = ""
        self._last_exception = ""
        self._last_stack_trace = ""
        self._last_test_results = []
        self._last_variables = {}
        self._passed_tests = 0
        self._all_tests_passed = False
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: CodingAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodingObservation:
        del kwargs
        self._state.step_count += 1
        reward = -0.05
        exec_timeout = min(timeout_s or EXECUTION_TIMEOUT_S, EXECUTION_TIMEOUT_S)

        if self._all_tests_passed:
            return self._build_observation(reward=reward, done=True)

        try:
            if action.action_type == "edit_line":
                if action.line_no is None:
                    raise ValueError("line_no required")
                self._replace_line(action.line_no, action.new_code or "")
            elif action.action_type == "edit_block":
                if self._replace_block(
                    action.start_line,
                    action.end_line,
                    action.new_code or "",
                ):
                    if action.start_line is not None and action.end_line is not None:
                        total_lines = max(1, len(self._code_lines))
                        change_ratio = (action.end_line - action.start_line + 1)*1.0 / total_lines
                        reward -= 0.1*change_ratio
                else:
                    reward -= 0.2
            elif action.action_type == "insert_line":
                self._insert_line(action.line_no or 1, action.code or "")
            elif action.action_type == "delete_line":
                self._delete_line(action.line_no or 1)
            elif action.action_type == "run_tests":
                reward += self._run_tests(exec_timeout)
            elif action.action_type == "inspect_variable":
                self._inspect_variable(action.var_name or "", exec_timeout)
            elif action.action_type == "get_stack_trace":   
                if not self._last_stack_trace:
                    reward += self._run_tests(exec_timeout)
        except Exception as exc:
            self._last_exception = str(exc)
            self._last_stack_trace = traceback.format_exc()
            reward -= 0.2

        done = self._all_tests_passed or self._state.step_count >= self.max_steps
        return self._build_observation(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    def _build_observation(self, reward: float, done: bool) -> CodingObservation:
        return CodingObservation(
            code=self._render_code(),
            stdout=self._last_stdout,
            stderr=self._last_stderr,
            exception=self._last_exception,
            stack_trace=self._last_stack_trace,
            test_results=self._last_test_results,
            variables=self._last_variables,
            step_count=self._state.step_count,
            reward=reward,
            done=done,
            metadata={
                "max_steps": self.max_steps,
                "tests_passed": self._passed_tests,
                "total_tests": len(self._test_cases),
                "difficulty": self._task.get("difficulty"),
                "task_name": self._task.get("task_name"),
                "visible_tests": self._test_cases,
            },
        )

    def _render_code(self) -> str:
        return "\n".join(self._code_lines) + "\n"

    def _replace_line(self, line_no: int, new_code: str) -> None:
        self._validate_line_number(line_no, allow_endpoint=False)
        self._code_lines[line_no - 1] = new_code

    def _insert_line(self, line_no: int, code: str) -> None:
        self._validate_line_number(line_no, allow_endpoint=True)
        self._code_lines.insert(line_no - 1, code)

    def _delete_line(self, line_no: int) -> None:
        self._validate_line_number(line_no, allow_endpoint=False)
        del self._code_lines[line_no - 1]

    def _replace_block(
        self,
        start_line: Optional[int],
        end_line: Optional[int],
        new_code: str,
    ) -> bool:
        if start_line is None or end_line is None:
            return False
        if start_line < 1 or end_line < start_line:
            return False
        if not self._code_lines:
            return False
        if start_line > len(self._code_lines) or end_line > len(self._code_lines):
            return False

        replacement = new_code.splitlines()
        self._code_lines[start_line-1 : end_line] = replacement
        return True

    def _validate_line_number(self, line_no: int, allow_endpoint: bool) -> None:
        max_line = len(self._code_lines) + (1 if allow_endpoint else 0)
        if line_no < 1 or line_no > max_line:
            raise ValueError(f"line_no must be between 1 and {max_line}")

    def _run_tests(self, timeout_s: float) -> float:
        previous_passed = self._passed_tests
        execution = self._execute_runner(self._build_test_runner(), timeout_s)

        self._last_stdout = execution["stdout"]
        self._last_stderr = execution["stderr"]

        if execution["status"] == "timeout":
            self._last_exception = "Execution timed out"
            self._last_stack_trace = ""
            self._last_test_results = self._timeout_results()
            self._passed_tests = 0
            self._all_tests_passed = False
            return -0.5

        payload = execution.get("payload") or {}
        self._last_exception = str(payload.get("exception") or "")
        self._last_stack_trace = str(payload.get("stack_trace") or "")
        self._last_test_results = [
            TestCaseResult(**result) for result in payload.get("test_results", [])
        ]
        self._passed_tests = sum(1 for result in self._last_test_results if result.passed)
        self._all_tests_passed = self._passed_tests == len(self._test_cases)

        reward = 0.0
        if self._all_tests_passed:
            reward += 1.0
        elif self._passed_tests > previous_passed:
            reward += 0.3

        if execution["returncode"] != 0 or self._last_exception:
            reward -= 0.2

        return reward

    def _inspect_variable(self, var_name: str, timeout_s: float) -> None:
        execution = self._execute_runner(self._build_inspection_runner(var_name), timeout_s)
        self._last_stdout = execution["stdout"]
        self._last_stderr = execution["stderr"]

        if execution["status"] == "timeout":
            self._last_exception = "Execution timed out"
            self._last_stack_trace = ""
            self._last_variables = {var_name: "TIMEOUT"}
            return

        payload = execution.get("payload") or {}
        self._last_exception = str(payload.get("exception") or "")
        self._last_stack_trace = str(payload.get("stack_trace") or "")
        variables = payload.get("variables") or {}
        self._last_variables = dict(variables)

    def _execute_runner(self, runner_source: str, timeout_s: float) -> Dict[str, Any]:
        temp_dir = os.path.join(TEMP_ROOT, str(uuid4()))
        os.makedirs(temp_dir, exist_ok=True)

        solution_path = os.path.join(temp_dir, "solution.py")
        runner_path = os.path.join(temp_dir, "runner.py")
        result_path = os.path.join(temp_dir, "result.json")

        try:
            with open(solution_path, "w", encoding="utf-8") as solution_file:
                solution_file.write(self._render_code())
            with open(runner_path, "w", encoding="utf-8") as runner_file:
                runner_file.write(runner_source)

            command = [sys.executable, "-I", runner_path]
            env = {
                "PYTHONIOENCODING": "utf-8",
                "OPENENV_RESULT_PATH": result_path,
                "OPENENV_SOLUTION_PATH": solution_path,
            }

            try:
                completed = subprocess.run(
                    command,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    env=env,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return {
                    "status": "timeout",
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                    "returncode": None,
                    "payload": None,
                }

            payload = None
            if os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as result_file:
                    payload = json.load(result_file)

            return {
                "status": "completed",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
                "payload": payload,
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _timeout_results(self) -> List[TestCaseResult]:
        return [
            TestCaseResult(
                input=case["input"],
                expected=case["expected"],
                actual="TIMEOUT",
                passed=False,
            )
            for case in self._test_cases
        ]

    def _build_test_runner(self) -> str:
        return textwrap.dedent(
            f"""
            import importlib.util
            import json
            import os
            import traceback

            RESULT_PATH = os.environ["OPENENV_RESULT_PATH"]
            SOLUTION_PATH = os.environ["OPENENV_SOLUTION_PATH"]
            TEST_CASES = {repr(self._test_cases)}

            def load_module():
                spec = importlib.util.spec_from_file_location("debug_solution", SOLUTION_PATH)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

            payload = {{
                "exception": "",
                "stack_trace": "",
                "test_results": [],
            }}

            try:
                module = load_module()
                for case in TEST_CASES:
                    try:
                        args = case["input"]
                        if not isinstance(args, list):
                            args = [args]
                        target = getattr(module, case["function_name"])
                        actual = target(*args)
                        payload["test_results"].append({{
                            "input": case["input"],
                            "expected": case["expected"],
                            "actual": actual,
                            "passed": actual == case["expected"],
                        }})
                    except Exception as exc:
                        payload["exception"] = str(exc)
                        payload["stack_trace"] = traceback.format_exc()
                        payload["test_results"].append({{
                            "input": case["input"],
                            "expected": case["expected"],
                            "actual": f"{{type(exc).__name__}}: {{exc}}",
                            "passed": False,
                        }})
            except Exception as exc:
                payload["exception"] = str(exc)
                payload["stack_trace"] = traceback.format_exc()
                for case in TEST_CASES:
                    payload["test_results"].append({{
                        "input": case["input"],
                        "expected": case["expected"],
                        "actual": f"{{type(exc).__name__}}: {{exc}}",
                        "passed": False,
                    }})

            with open(RESULT_PATH, "w", encoding="utf-8") as result_file:
                json.dump(payload, result_file)
            """
        )

    def _build_inspection_runner(self, var_name: str) -> str:
        inspect_case = self._select_inspection_case()
        inspect_input = inspect_case["input"]
        return textwrap.dedent(
            f"""
            import importlib.util
            import json
            import os
            import sys
            import traceback

            RESULT_PATH = os.environ["OPENENV_RESULT_PATH"]
            SOLUTION_PATH = os.environ["OPENENV_SOLUTION_PATH"]
            FUNCTION_NAME = {repr(inspect_case["function_name"])}
            VAR_NAME = {repr(var_name)}
            TEST_INPUT = {repr(inspect_input)}

            def load_module():
                spec = importlib.util.spec_from_file_location("debug_solution", SOLUTION_PATH)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

            captured = {{}}
            payload = {{
                "exception": "",
                "stack_trace": "",
                "variables": {{}},
            }}

            def tracer(frame, event, arg):
                if frame.f_code.co_name == FUNCTION_NAME and VAR_NAME in frame.f_locals:
                    captured[VAR_NAME] = repr(frame.f_locals[VAR_NAME])
                return tracer

            try:
                module = load_module()
                target = getattr(module, FUNCTION_NAME)
                sys.settrace(tracer)
                try:
                    args = TEST_INPUT
                    if not isinstance(args, list):
                        args = [args]
                    target(*args)
                finally:
                    sys.settrace(None)
                payload["variables"] = {{VAR_NAME: captured.get(VAR_NAME, "UNAVAILABLE")}}
            except Exception as exc:
                sys.settrace(None)
                payload["exception"] = str(exc)
                payload["stack_trace"] = traceback.format_exc()
                payload["variables"] = {{VAR_NAME: captured.get(VAR_NAME, "UNAVAILABLE")}}

            with open(RESULT_PATH, "w", encoding="utf-8") as result_file:
                json.dump(payload, result_file)
            """
        )

    def _select_inspection_case(self) -> Dict[str, Any]:
        for result in self._last_test_results:
            if not result.passed:
                for case in self._test_cases:
                    if case["input"] == result.input:
                        return case
        return self._test_cases[0]
