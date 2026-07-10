# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for GPUModelRunner._update_streaming_request function."""

from unittest.mock import Mock

import pytest

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def mock_model_runner_with_input_batch():
    """Create a mock GPUModelRunner with a real InputBatch for e2e testing."""

    runner = Mock(spec=GPUModelRunner)
    runner.uses_mrope = False
    runner.requests = {}
    runner.max_num_reqs = 10
    runner.max_model_len = 1024
    # Mirrors gpu_model_runner.py:825 self.late_interaction_runner =
    # LateInteractionRunner(); an instance attr the spec-based Mock lacks.
    runner.late_interaction_runner = Mock()
    # Sync-consumer metadata stashes the runner inherits from
    # CaptureRunnerMixin.__init__ (capture_runner_mixin.py:109 and :113);
    # _update_streaming_request refreshes them, but Mock(spec=...) never runs
    # __init__ so the instance attrs are absent.
    runner._sync_conversation_ids = {}
    runner._sync_steering_gates = {}

    # Create a real InputBatch for e2e testing
    runner.input_batch = InputBatch(
        max_num_reqs=10,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        device="cpu",
        pin_memory=False,
        vocab_size=32000,
        block_sizes=[16],
        kernel_block_sizes=[16],
        logitsprocs=None,
        is_pooling_model=False,
    )
    return runner


def test_e2e_streaming_request_update_basic_flow(mock_model_runner_with_input_batch):
    """Test that streaming session are updated correctly.

    This test validates that when a streaming session is updated with new prompt tokens:
    1. The request is removed from InputBatch before updating (avoids duplication)
    2. Request state fields are updated correctly
    3. output_token_ids is cleared (intermediate outputs are now in prompt_token_ids)
    """
    runner = mock_model_runner_with_input_batch
    req_id = "streaming_req_0"

    # Step 1: Create initial request state with some computed tokens
    initial_req_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.5),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=3,
        output_token_ids=[10, 11],  # Generated 2 tokens
    )
    runner.requests[req_id] = initial_req_state

    # Add request to InputBatch
    runner.input_batch.add_request(initial_req_state)
    assert req_id in runner.input_batch.req_id_to_index

    # Step 2: Create new request data with extended prompt
    # The scheduler has already set prompt_token_ids to the full sequence
    # (original prompt + intermediate outputs + new prompt)
    new_req_data = Mock()
    new_req_data.prompt_token_ids = [
        1,
        2,
        3,
        10,
        4,
        5,
    ]  # Full sequence with intermediate output (10)
    new_req_data.mm_features = []
    new_req_data.prompt_embeds = None
    new_req_data.sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
    new_req_data.pooling_params = None
    new_req_data.block_ids = ([0, 1],)
    new_req_data.num_computed_tokens = 4  # 3 original prompt + 1 intermediate output

    # Step 3: Update the request
    updated_req_state = GPUModelRunner._update_streaming_request(
        runner, req_id, new_req_data
    )

    # Step 4: Verify the request state was updated correctly
    assert updated_req_state.prompt_token_ids == [1, 2, 3, 10, 4, 5]
    assert updated_req_state.num_computed_tokens == 4
    assert updated_req_state.sampling_params.temperature == 0.8
    assert updated_req_state.sampling_params.max_tokens == 50
    assert updated_req_state.block_ids == ([0, 1],)

    # Verify output_token_ids were cleared
    # (intermediate outputs are now in prompt_token_ids)
    assert updated_req_state.output_token_ids == []

    # Verify the same object is returned
    assert runner.requests[req_id] is updated_req_state

    # Verify request was removed from InputBatch during update (avoids duplication)
    assert req_id not in runner.input_batch.req_id_to_index


def test_e2e_streaming_with_multimodal_features(mock_model_runner_with_input_batch):
    """Test that streaming session with multimodal features are updated correctly.

    This test validates that when a streaming session with mm features is updated:
    1. The request is removed from InputBatch before updating (avoids duplication)
    2. Multimodal features from both requests are preserved and merged correctly
    3. New prompt tokens (including intermediate outputs) are appended correctly
    4. output_token_ids is cleared (intermediate outputs are now in prompt_token_ids)
    """
    runner = mock_model_runner_with_input_batch
    req_id = "streaming_mm_req_0"

    # Step 1: Create initial request state with one multimodal feature
    mm_feature_1 = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        modality="audio",
        identifier="audio_1",
        mm_position=PlaceholderRange(offset=2, length=10),
    )

    initial_req_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2] + [0] * 10 + [3, 4],  # 2 + 10 (mm) + 2 = 14 tokens
        mm_features=[mm_feature_1],
        sampling_params=SamplingParams(),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=14,
        output_token_ids=[100],  # Generated 1 token
    )
    runner.requests[req_id] = initial_req_state

    # Add request to InputBatch
    runner.input_batch.add_request(initial_req_state)
    assert req_id in runner.input_batch.req_id_to_index

    # Step 2: Create new request data with additional multimodal feature
    # The scheduler has already set prompt_token_ids to the full sequence
    # (original prompt + intermediate outputs + new prompt with new multimodal feature)
    mm_feature_2 = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        modality="audio",
        identifier="audio_2",
        mm_position=PlaceholderRange(offset=15, length=5),
    )

    new_req_data = Mock()
    # Full sequence: [1, 2] + [0]*10 + [3, 4] + [100] + [0]*5 + [5] = 21 tokens
    new_req_data.prompt_token_ids = [1, 2] + [0] * 10 + [3, 4, 100] + [0] * 5 + [5]
    new_req_data.mm_features = [mm_feature_1, mm_feature_2]
    new_req_data.prompt_embeds = None
    new_req_data.sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
    new_req_data.pooling_params = None
    new_req_data.block_ids = ([0, 1],)
    new_req_data.num_computed_tokens = 14  # 14 tokens from initial request

    # Step 3: Update the request
    updated_req_state = GPUModelRunner._update_streaming_request(
        runner, req_id, new_req_data
    )

    # Step 4: Verify the request state was updated correctly
    # Verify multimodal features are preserved
    assert len(updated_req_state.mm_features) == 2
    assert updated_req_state.mm_features[0] == mm_feature_1
    assert updated_req_state.mm_features[1] == mm_feature_2

    # Verify prompt tokens include intermediate output (100) and new tokens
    # Initial: 2 + 10 (mm1) + 2 = 14 tokens
    # New: 2 + 10 (mm1) + 2 + 1 (output 100) + 5 (mm2) + 1 = 21 tokens
    assert len(updated_req_state.prompt_token_ids) == 21
    assert updated_req_state.prompt_token_ids == [1, 2] + [0] * 10 + [3, 4, 100] + [
        0
    ] * 5 + [5]

    # Verify output_token_ids were cleared
    # (intermediate outputs are now in prompt_token_ids)
    assert updated_req_state.output_token_ids == []

    # Verify other parameters were updated
    assert updated_req_state.num_computed_tokens == 14
    assert updated_req_state.sampling_params.temperature == 0.7
    assert updated_req_state.sampling_params.max_tokens == 30
    assert updated_req_state.block_ids == ([0, 1],)

    # Verify the same object is returned
    assert runner.requests[req_id] is updated_req_state

    # Verify request was removed from InputBatch during update (avoids duplication)
    assert req_id not in runner.input_batch.req_id_to_index


def test_streaming_re_add_refreshes_sync_metadata_stashes():
    """A streaming re-add must refresh the sync-consumer metadata stashes.

    Regression for the v1/v2 drift where ``_update_streaming_request`` never
    refreshed ``_sync_conversation_ids`` / ``_sync_steering_gates``, so a
    streaming continuation that changed the conversation id or declarative
    gates kept serving the prior chunk's stale values (v2 already refreshes
    them unconditionally in ``_capture_add_request``).
    """
    from vllm.v1.request_metadata import RequestMetadata
    from vllm.v1.steering_schema import build_steering_gates, resolve_gates_safe

    req_id = "streaming_meta_req"

    runner = Mock(spec=GPUModelRunner)
    runner.uses_mrope = False
    runner.late_interaction_runner = Mock()
    runner.input_batch = InputBatch(
        max_num_reqs=10,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        device="cpu",
        pin_memory=False,
        vocab_size=32000,
        block_sizes=[16],
        kernel_block_sizes=[16],
        logitsprocs=None,
        is_pooling_model=False,
    )
    # Stale stashes left over from the prior chunk's registration.
    runner._sync_conversation_ids = {req_id: "conv-old"}
    runner._sync_steering_gates = {req_id: ["STALE"]}

    req_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=3,
        output_token_ids=[9],
    )
    runner.requests = {req_id: req_state}
    runner.input_batch.add_request(req_state)

    gates = build_steering_gates(
        [
            {
                "when": {"kind": "always"},
                "scope": "rest_of_request",
                "apply": {"kind": "attenuate", "strength": 0.25},
            }
        ],
        None,
    )

    new_req_data = Mock()
    new_req_data.prompt_token_ids = [1, 2, 3, 9, 4]
    new_req_data.mm_features = []
    new_req_data.prompt_embeds = None
    new_req_data.sampling_params = SamplingParams(max_tokens=8)
    new_req_data.pooling_params = None
    new_req_data.block_ids = ([0, 1],)
    new_req_data.num_computed_tokens = 4
    new_req_data.prefill_steering_config_hash = 0
    new_req_data.decode_steering_config_hash = 0
    new_req_data.request_metadata = RequestMetadata(
        conversation_id="conv-new", steering=gates
    )

    GPUModelRunner._update_streaming_request(runner, req_id, new_req_data)

    # Conversation id refreshed to the continuation's value (not stale).
    assert runner._sync_conversation_ids[req_id] == "conv-new"
    # Declarative gates refreshed to the resolved continuation gates.
    refreshed = runner._sync_steering_gates[req_id]
    assert refreshed != ["STALE"]
    assert refreshed is not None
    assert len(refreshed) == len(resolve_gates_safe(gates, req_id))
