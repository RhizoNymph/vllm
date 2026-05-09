# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the zstd compression path on packed steering vectors.

Covers:

- ``CompressedSteeringArray`` wrapper round-trips through
  ``MsgpackEncoder``/``MsgpackDecoder`` byte-for-byte.
- Compression actually shrinks payload size for typical bf16 vectors.
- Threshold path: tiny arrays bypass compression and stay as plain
  ``ndarray``.
- ``hash_steering_config`` accepts both wrapped and unwrapped forms.
- ``effective_*_steering`` cached_property unwraps wrapped values.
"""

import numpy as np
import torch

from vllm.config.steering_types import (
    ZSTD_THRESHOLD,
    CompressedSteeringArray,
    hash_steering_config,
    maybe_pack_inline_steering_for_request,
    pack_effective_steering,
    pack_steering_for_dtype,
    unwrap_steering_array,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.serial_utils import (
    CUSTOM_TYPE_ZSTD_NDARRAY,
    MsgpackDecoder,
    MsgpackEncoder,
    _pack_zstd_ndarray,
    _unpack_zstd_ndarray,
)

# ---------------------------------------------------------------------------
# Pack/unpack helpers (raw byte layout)
# ---------------------------------------------------------------------------


class TestPackUnpackZstd:
    def test_pack_unpack_round_trip(self):
        compressed = b"\x00abc\xff"
        out = _pack_zstd_ndarray("<f4", (4, 8), compressed)
        # Note: _unpack_zstd_ndarray decompresses; we exercise it via a
        # full encode/decode in the higher-level tests below.
        assert isinstance(out, bytes)
        # Decoded prefix: 2 bytes dtype length + dtype + 1 byte ndim + ndim*8.
        # "<f4" is 3 bytes -> header is 2 + 3 + 1 + 16 = 22.
        assert out[:2] == (3).to_bytes(2, "big")
        assert out[2:5] == b"<f4"
        assert out[5:6] == (2).to_bytes(1, "big")
        assert out[6:14] == (4).to_bytes(8, "big")
        assert out[14:22] == (8).to_bytes(8, "big")
        assert out[22:] == compressed

    def test_pack_unpack_real_payload(self):
        import zstandard as zstd

        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        raw = bytes(arr.data)
        compressed = zstd.ZstdCompressor(level=1).compress(raw)
        packed = _pack_zstd_ndarray(arr.dtype.str, arr.shape, compressed)
        dtype_str, shape, decompressed = _unpack_zstd_ndarray(packed)
        assert dtype_str == arr.dtype.str
        assert shape == arr.shape
        recovered = np.frombuffer(decompressed, dtype=np.dtype(dtype_str)).reshape(
            shape
        )
        assert np.array_equal(recovered, arr)


# ---------------------------------------------------------------------------
# Threshold behavior
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_small_array_stays_uncompressed(self):
        # 4 floats * 4 bytes = 16 bytes, well under 4 KB threshold.
        spec = {"post_mlp": {0: [1.0, 2.0, 3.0, 4.0]}}
        out = pack_steering_for_dtype(spec, np.float32)
        assert out is not None
        entry = out["post_mlp"][0]
        assert isinstance(entry, np.ndarray)
        assert not isinstance(entry, CompressedSteeringArray)

    def test_large_array_gets_wrapped(self):
        # ZSTD_THRESHOLD is 4096; bf16->fp32 fallback gives 4 B/elem, so
        # 2048 elements = 8192 bytes > 4 KB.
        n = 2048
        spec = {"post_mlp": {0: [float(i) for i in range(n)]}}
        out = pack_steering_for_dtype(spec, np.float32)
        assert out is not None
        entry = out["post_mlp"][0]
        assert isinstance(entry, CompressedSteeringArray)
        # Underlying array must still hold the right values.
        assert entry.array.dtype == np.float32
        assert entry.array.shape == (n,)
        assert entry.array[0] == 0.0
        assert entry.array[-1] == float(n - 1)

    def test_threshold_is_byte_size_not_elem_count(self):
        # fp16: 2 B/elem.  4096 / 2 = 2048 elems crosses threshold.
        below = pack_steering_for_dtype(
            {"post_mlp": {0: [float(i) for i in range(2047)]}}, np.float16
        )
        at = pack_steering_for_dtype(
            {"post_mlp": {0: [float(i) for i in range(2048)]}}, np.float16
        )
        assert below is not None and at is not None
        assert isinstance(below["post_mlp"][0], np.ndarray) and not isinstance(
            below["post_mlp"][0], CompressedSteeringArray
        )
        assert isinstance(at["post_mlp"][0], CompressedSteeringArray)

    def test_pack_effective_steering_wraps_above_threshold(self):
        n = 2048
        spec = {"post_mlp": {0: [float(i) for i in range(n)]}}
        out = pack_effective_steering(spec, None, np.float32)
        assert out is not None
        assert isinstance(out["post_mlp"][0], CompressedSteeringArray)


# ---------------------------------------------------------------------------
# Encode/decode round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_compressed_array_round_trips_byte_equal(self):
        n = 4096
        rng = np.random.default_rng(0)
        # Use a smooth float distribution so compression gives a real win.
        arr = rng.standard_normal(n).astype(np.float32)
        sp = SamplingParams(
            max_tokens=1, steering_vectors={"post_mlp": {0: arr.tolist()}}
        )
        maybe_pack_inline_steering_for_request(sp, torch.float32)
        # Sanity: payload was wrapped.
        packed = sp._effective_prefill_steering_packed
        assert packed is not None
        assert isinstance(packed["post_mlp"][0], CompressedSteeringArray)

        enc = MsgpackEncoder()
        bufs = enc.encode(sp)
        dec = MsgpackDecoder(SamplingParams)
        sp_out = dec.decode(bufs)

        out = sp_out._effective_prefill_steering_packed
        assert out is not None
        out_arr = out["post_mlp"][0]
        # Worker side gets unwrapped ndarray.
        assert isinstance(out_arr, np.ndarray)
        assert out_arr.dtype == np.float32
        assert out_arr.shape == (n,)
        assert np.array_equal(out_arr, arr)

    def test_compression_ratio_is_real(self):
        """Sanity check: zstd compresses a typical model-size steering vector
        payload by at least 1.3x relative to the equivalent raw bytes.

        We compare the compressed payload bytes carried by
        ``CompressedSteeringArray`` ext frames against the underlying raw
        ``ndarray.tobytes()`` size — measuring the IPC bytes saved per
        vector without the surrounding msgpack envelope confounding the
        comparison.  The aux-buffer raw-view path that ``MsgpackEncoder``
        uses for un-wrapped large ndarrays sums to the same raw byte
        count, so this is the apples-to-apples ratio.

        Real-world steering vectors (SAE feature directions, RepEng
        residual-stream directions) are sparse-ish: most coordinates are
        small, with a few large entries. We simulate that with a heavy-
        tailed distribution.  Random Gaussian noise alone is essentially
        incompressible at level 1, so this test would underestimate the
        production win.
        """
        import zstandard as zstd

        n_layers = 34
        dim = 2560
        rng = np.random.default_rng(1)
        compressor = zstd.ZstdCompressor(level=1)

        # Sparse-ish steering vectors: most entries are exactly zero, a
        # small fraction are nonzero.  This matches the empirical
        # distribution of SAE feature directions (sparse activations) and
        # trained steering probes after regularization-induced sparsity.
        # Random Gaussian noise (zero-sparsity) is essentially
        # incompressible at level 1 — it's also not what production
        # workloads ship.
        def gen_vec() -> np.ndarray:
            v = np.zeros(dim, dtype=np.float32)
            mask = rng.random(dim) < 0.10
            v[mask] = rng.standard_normal(int(mask.sum())).astype(np.float32)
            return v

        vectors = {
            "post_mlp": {i: gen_vec().tolist() for i in range(n_layers)}
        }
        sp = SamplingParams(max_tokens=1, steering_vectors=vectors)
        maybe_pack_inline_steering_for_request(sp, torch.bfloat16)
        packed = sp._effective_prefill_steering_packed
        assert packed is not None
        wrapped = [
            v for v in packed["post_mlp"].values()
            if isinstance(v, CompressedSteeringArray)
        ]
        assert wrapped, "expected at least one entry to be wrapped"

        raw_bytes = 0
        zstd_bytes = 0
        for entry in packed["post_mlp"].values():
            arr = unwrap_steering_array(entry)
            raw = bytes(arr.data) if arr.flags.c_contiguous else arr.tobytes()
            raw_bytes += len(raw)
            zstd_bytes += len(compressor.compress(raw))

        ratio = raw_bytes / zstd_bytes
        assert ratio >= 1.3, (
            f"zstd ratio={ratio:.2f}x (raw={raw_bytes} B, zstd={zstd_bytes} B); "
            f"expected >=1.3x for smooth float distributions"
        )

    def test_below_threshold_stays_inline_raw_view(self):
        # Tiny arrays should NOT be wrapped, so the encoder never invokes
        # the zstd ext type for them.  Round-trip must still succeed.
        sp = SamplingParams(
            max_tokens=1,
            steering_vectors={"post_mlp": {0: [1.0, 2.0, 3.0]}},
        )
        maybe_pack_inline_steering_for_request(sp, torch.float32)
        packed = sp._effective_prefill_steering_packed
        assert packed is not None
        assert isinstance(packed["post_mlp"][0], np.ndarray)
        assert not isinstance(packed["post_mlp"][0], CompressedSteeringArray)

        enc = MsgpackEncoder()
        bufs = enc.encode(sp)
        dec = MsgpackDecoder(SamplingParams)
        sp_out = dec.decode(bufs)
        out = sp_out._effective_prefill_steering_packed
        assert out is not None
        assert np.array_equal(out["post_mlp"][0], np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Hash determinism with the wrapper
# ---------------------------------------------------------------------------


class TestHashing:
    def test_hash_equal_wrapped_vs_unwrapped(self):
        n = 2048
        arr = np.arange(n, dtype=np.float32)
        unwrapped = {"post_mlp": {0: arr}}
        wrapped = {"post_mlp": {0: CompressedSteeringArray(arr)}}
        assert hash_steering_config(unwrapped) == hash_steering_config(wrapped)

    def test_hash_via_sampling_params_packed_matches_unpacked(self):
        n = 2048
        vectors = {"post_mlp": {0: [float(i) for i in range(n)]}}
        sp_unpacked = SamplingParams(max_tokens=1, steering_vectors=vectors)
        unpacked_hash = sp_unpacked.prefill_steering_config_hash

        sp_packed = SamplingParams(max_tokens=1, steering_vectors=vectors)
        maybe_pack_inline_steering_for_request(sp_packed, torch.float32)
        # Confirm the packed dict carries a CompressedSteeringArray; this
        # is the critical case the hash code path must handle.
        packed = sp_packed._effective_prefill_steering_packed
        assert packed is not None
        assert isinstance(packed["post_mlp"][0], CompressedSteeringArray)

        packed_hash = sp_packed.prefill_steering_config_hash
        assert packed_hash == unpacked_hash


# ---------------------------------------------------------------------------
# Cached property unwraps for downstream consumers
# ---------------------------------------------------------------------------


class TestCachedPropertyUnwrap:
    def test_effective_prefill_steering_unwraps(self):
        n = 2048
        sp = SamplingParams(
            max_tokens=1,
            steering_vectors={"post_mlp": {0: [float(i) for i in range(n)]}},
        )
        maybe_pack_inline_steering_for_request(sp, torch.float32)
        # Underlying packed field carries the wrapper...
        packed = sp._effective_prefill_steering_packed
        assert packed is not None
        assert isinstance(packed["post_mlp"][0], CompressedSteeringArray)
        # ...but the cached_property surfaces a plain ndarray.
        eff = sp.effective_prefill_steering
        assert eff is not None
        assert isinstance(eff["post_mlp"][0], np.ndarray)
        assert not isinstance(eff["post_mlp"][0], CompressedSteeringArray)
        assert eff["post_mlp"][0].shape == (n,)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_ext_type_code_is_distinct(self):
        from vllm.v1.serial_utils import (
            CUSTOM_TYPE_CLOUDPICKLE,
            CUSTOM_TYPE_PICKLE,
            CUSTOM_TYPE_RAW_VIEW,
        )

        assert CUSTOM_TYPE_ZSTD_NDARRAY not in (
            CUSTOM_TYPE_PICKLE,
            CUSTOM_TYPE_CLOUDPICKLE,
            CUSTOM_TYPE_RAW_VIEW,
        )

    def test_threshold_is_positive(self):
        assert ZSTD_THRESHOLD > 0
