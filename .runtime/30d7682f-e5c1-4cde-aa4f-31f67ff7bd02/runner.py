
import importlib.util
import json
import os
import sys
import traceback

RESULT_PATH = os.environ["OPENENV_RESULT_PATH"]
SOLUTION_PATH = os.environ["OPENENV_SOLUTION_PATH"]
FUNCTION_NAME = 'summarize_numbers'
VAR_NAME = 'total'
TEST_INPUT = [2, 4, 6]

def load_module():
    spec = importlib.util.spec_from_file_location("debug_solution", SOLUTION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

captured = {}
payload = {
    "exception": "",
    "stack_trace": "",
    "variables": {},
}

def tracer(frame, event, arg):
    if frame.f_code.co_name == FUNCTION_NAME and VAR_NAME in frame.f_locals:
        captured[VAR_NAME] = repr(frame.f_locals[VAR_NAME])
    return tracer

try:
    module = load_module()
    target = getattr(module, FUNCTION_NAME)
    sys.settrace(tracer)
    try:
        target(TEST_INPUT)
    finally:
        sys.settrace(None)
    payload["variables"] = {VAR_NAME: captured.get(VAR_NAME, "UNAVAILABLE")}
except Exception as exc:
    sys.settrace(None)
    payload["exception"] = str(exc)
    payload["stack_trace"] = traceback.format_exc()
    payload["variables"] = {VAR_NAME: captured.get(VAR_NAME, "UNAVAILABLE")}

with open(RESULT_PATH, "w", encoding="utf-8") as result_file:
    json.dump(payload, result_file)
