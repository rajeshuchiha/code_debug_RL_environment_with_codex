# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Coding Env debugging environment."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CodingAction(Action):
    """Structured action for the code debugging environment."""

    action_type: Literal[
        "edit_line",
        "insert_line",
        "delete_line",
        "run_tests",
        "inspect_variable",
        "get_stack_trace",
        "edit_block",
    ] = Field(..., description="The action to execute.")
    line_no: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based line number for edit, insert, or delete operations.",
    )
    new_code: Optional[str] = Field(
        default=None,
        description="Replacement code for edit_line.",
    )
    code: Optional[str] = Field(
        default=None,
        description="Code to insert for insert_line.",
    )
    var_name: Optional[str] = Field(
        default=None,
        description="Variable name to inspect for inspect_variable.",
    )
    start_line: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based start line for edit_block.",
    )
    end_line: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based inclusive end line for edit_block.",
    )

    @model_validator(mode="before")
    @classmethod
    def flatten_parameters(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("action_type") != "edit_block":
            return data

        values = dict(data)
        parameters = values.pop("parameters", None)
        if isinstance(parameters, dict):
            for key in ("start_line", "end_line", "new_code"):
                if key in parameters and key not in values:
                    values[key] = parameters[key]
            if "new_code_block" in parameters and "new_code" not in values:
                values["new_code"] = parameters["new_code_block"]
        return values

    @model_validator(mode="after")
    def validate_payload(self) -> "CodingAction":
        line_actions = {"edit_line", "insert_line", "delete_line"}
        if self.action_type in line_actions and self.line_no is None:
            raise ValueError("line_no is required for line editing actions")
        if self.action_type == "edit_line" and self.new_code is None:
            raise ValueError("new_code is required for edit_line")
        if self.action_type == "insert_line" and self.code is None:
            raise ValueError("code is required for insert_line")
        if self.action_type == "inspect_variable" and not self.var_name:
            raise ValueError("var_name is required for inspect_variable")
        if self.action_type == "edit_block":
            if self.start_line is None:
                raise ValueError("start_line is required for edit_block")
            if self.end_line is None:
                raise ValueError("end_line is required for edit_block")
            if self.end_line < self.start_line:
                raise ValueError("end_line must be greater than or equal to start_line")
            if self.new_code is None:
                raise ValueError("new_code is required for edit_block")
        return self


class TestCaseResult(BaseModel):
    """Per-test execution result."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    input: Any = Field(..., description="Input used for the test case.")
    expected: Any = Field(..., description="Expected result for the test case.")
    actual: Any = Field(default=None, description="Actual result or error string.")
    passed: bool = Field(default=False, description="Whether the test passed.")


class CodingObservation(Observation):
    """Observation for the debugging environment."""

    code: str = Field(default="", description="Current source code under repair.")
    stdout: str = Field(default="", description="Captured stdout from the last execution.")
    stderr: str = Field(default="", description="Captured stderr from the last execution.")
    exception: str = Field(default="", description="Last exception message, if any.")
    stack_trace: str = Field(default="", description="Last stack trace, if any.")
    test_results: List[TestCaseResult] = Field(
        default_factory=list,
        description="Structured results for each test case.",
    )
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Variables captured through inspection.",
    )
    step_count: int = Field(default=0, description="Number of steps taken in the episode.")
