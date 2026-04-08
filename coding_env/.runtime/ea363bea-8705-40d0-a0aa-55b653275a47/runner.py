
import importlib.util
import json
import os
import traceback

RESULT_PATH = os.environ["OPENENV_RESULT_PATH"]
SOLUTION_PATH = os.environ["OPENENV_SOLUTION_PATH"]
TEST_CASES = [{"input": [2, 4, 6], "expected": {"total": 12, "average": 4.0, "min": 2, "max": 6}}, {"input": [5], "expected": {"total": 5, "average": 5.0, "min": 5, "max": 5}}, {"input": [], "expected": {"total": 0, "average": 0, "min": null, "max": null}}]
FUNCTION_NAME = "summarize_numbers"

def load_module():
    spec = importlib.util.spec_from_file_location("debug_solution", SOLUTION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

payload = {
    "exception": "",
    "stack_trace": "",
    "test_results": [],
}

try:
    module = load_module()
    target = getattr(module, FUNCTION_NAME)
    for case in TEST_CASES:
        try:
            actual = target(case["input"])
            payload["test_results"].append({
                "input": case["input"],
                "expected": case["expected"],
                "actual": actual,
                "passed": actual == case["expected"],
            })
        except Exception as exc:
            payload["exception"] = str(exc)
            payload["stack_trace"] = traceback.format_exc()
            payload["test_results"].append({
                "input": case["input"],
                "expected": case["expected"],
                "actual": f"{type(exc).__name__}: {exc}",
                "passed": False,
            })
except Exception as exc:
    payload["exception"] = str(exc)
    payload["stack_trace"] = traceback.format_exc()
    for case in TEST_CASES:
        payload["test_results"].append({
            "input": case["input"],
            "expected": case["expected"],
            "actual": f"{type(exc).__name__}: {exc}",
            "passed": False,
        })

with open(RESULT_PATH, "w", encoding="utf-8") as result_file:
    json.dump(payload, result_file)
