# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="dummy_capture_consumer",
    version="0.1",
    packages=["dummy_capture_consumer"],
    entry_points={
        "vllm.capture_consumers": [
            "dummy_capture_consumer = dummy_capture_consumer.dummy_capture_consumer:DummyCaptureConsumer"  # noqa
        ]
    },
)
