# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the code debugging environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CodingAction, CodingObservation


class CodingEnv(EnvClient[CodingAction, CodingObservation, State]):
    """Typed client for the debugging environment."""

    def _step_payload(self, action: CodingAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[CodingObservation]:
        obs_data = payload.get("observation", {})
        observation = CodingObservation(
            code=obs_data.get("code", ""),
            stdout=obs_data.get("stdout", ""),
            stderr=obs_data.get("stderr", ""),
            exception=obs_data.get("exception", ""),
            stack_trace=obs_data.get("stack_trace", ""),
            test_results=obs_data.get("test_results", []),
            variables=obs_data.get("variables", {}),
            step_count=obs_data.get("step_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
