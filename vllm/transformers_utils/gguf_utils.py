# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF utility functions."""

from functools import cache
from os import PathLike
from pathlib import Path

import gguf
import regex as re
from gguf.constants import Keys, VisionProjectorType
from gguf.quants import GGMLQuantizationType
from transformers import Gemma3Config, PretrainedConfig, SiglipVisionConfig

from vllm.logger import init_logger

from .repo_utils import list_filtered_repo_files

logger = init_logger(__name__)


def _register_gemma4_gguf_support() -> None:
    """Teach transformers' GGUF loader about the gemma4 architecture.

    Transformers ships GGUF_CONFIG_MAPPING with entries for gemma2/gemma3
    but not gemma4, so load_gguf_checkpoint raises before vLLM ever sees
    the file. We patch the mapping in-place and wrap load_gguf_checkpoint
    to:
      * rename model_type "gemma4" -> "gemma4_text" (mirrors gemma3 path)
      * reduce per-layer head_count_kv lists into the scalar fields that
        Gemma4TextConfig expects (num_key_value_heads for sliding layers,
        num_global_key_value_heads for full-attention layers)
      * derive layer_types from attention.sliding_window_pattern
      * build rope_parameters from rope.freq_base{,_swa}
      * split key_length{,_swa} into head_dim / global_head_dim
    """
    from transformers import modeling_gguf_pytorch_utils as _mgu
    from transformers.integrations import ggml as _ggml

    if "gemma4" in _ggml.GGUF_CONFIG_MAPPING:
        return

    _ggml.GGUF_CONFIG_MAPPING["gemma4"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "attention.head_count": "num_attention_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.sliding_window": "sliding_window",
        "attention.shared_kv_layers": "num_kv_shared_layers",
        # NOTE: final_logit_softcapping intentionally not mapped. llama.cpp's
        # gemma4 GGUFs carry a value of 30.0 inherited from gemma2/3, but
        # the HF Gemma4TextConfig defaults to None and applying it here
        # saturates all token logits to ±30 (hidden state RMS is large), so
        # softmax collapses to ~uniform. Leave it unset.
        "embedding_length_per_layer_input": "hidden_size_per_layer_input",
        "vocab_size": "vocab_size",
    }
    _mgu.GGUF_SUPPORTED_ARCHITECTURES.append("gemma4")
    # convert_gguf_tokenizer keys by the raw GGUF architecture string
    # ("gemma4"), while elsewhere we use the renamed model_type
    # ("gemma4_text"). Register both so either lookup path works.
    _ggml.GGUF_TO_FAST_CONVERTERS["gemma4"] = _ggml.GGUFGemmaConverter
    _ggml.GGUF_TO_FAST_CONVERTERS["gemma4_text"] = _ggml.GGUFGemmaConverter

    _orig_load = _mgu.load_gguf_checkpoint

    def _patched_load(*args, **kwargs):
        parsed = _orig_load(*args, **kwargs)
        cfg = parsed.get("config", {})
        if cfg.get("model_type") != "gemma4":
            return parsed
        gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path")
        cfg["model_type"] = "gemma4_text"
        if not cfg.get("architectures"):
            cfg["architectures"] = ["Gemma4ForCausalLM"]
        reader = gguf.GGUFReader(gguf_path)
        fields = reader.fields
        pat_field = fields.get("gemma4.attention.sliding_window_pattern")
        nkv_field = fields.get("gemma4.attention.head_count_kv")
        layer_pat = None
        if pat_field is not None:
            layer_pat = [
                bool(pat_field.parts[i].tolist()[0]) for i in pat_field.data
            ]
            cfg["layer_types"] = [
                "sliding_attention" if p else "full_attention" for p in layer_pat
            ]
        if nkv_field is not None:
            nkv_list = [
                int(nkv_field.parts[i].tolist()[0]) for i in nkv_field.data
            ]
            if layer_pat is not None and len(layer_pat) == len(nkv_list):
                sliding = {v for v, p in zip(nkv_list, layer_pat) if p}
                full = {v for v, p in zip(nkv_list, layer_pat) if not p}
                if len(sliding) == 1:
                    cfg["num_key_value_heads"] = sliding.pop()
                if len(full) == 1:
                    cfg["num_global_key_value_heads"] = full.pop()
            elif isinstance(cfg.get("num_key_value_heads"), list):
                cfg["num_key_value_heads"] = max(cfg["num_key_value_heads"])
        key_swa = fields.get("gemma4.attention.key_length_swa")
        key_full = fields.get("gemma4.attention.key_length")
        if key_swa is not None:
            cfg["head_dim"] = int(key_swa.parts[key_swa.data[0]].tolist()[0])
        if key_full is not None:
            cfg["global_head_dim"] = int(
                key_full.parts[key_full.data[0]].tolist()[0]
            )
        # Detect attention_k_eq_v: if any full-attention block is missing
        # an attn_v tensor, the checkpoint uses shared K/V weights for
        # full-attention layers.
        if layer_pat is not None:
            tensor_names = {t.name for t in reader.tensors}
            full_blocks_missing_v = any(
                not p and f"blk.{i}.attn_v.weight" not in tensor_names
                for i, p in enumerate(layer_pat)
            )
            if full_blocks_missing_v:
                cfg["attention_k_eq_v"] = True
        rope_swa = fields.get("gemma4.rope.freq_base_swa")
        rope_full = fields.get("gemma4.rope.freq_base")
        rope_parameters: dict = {}
        if rope_swa is not None:
            rope_parameters["sliding_attention"] = {
                "rope_type": "default",
                "rope_theta": float(
                    rope_swa.parts[rope_swa.data[0]].tolist()[0]
                ),
            }
        if rope_full is not None:
            # Verified against the GGUF's rope_freqs.weight: only the first
            # head_dim/4 frequency pairs are active (= partial_rotary_factor
            # 0.25 with head_dim=512 → 64 active angles, 192 zero-padded).
            rope_parameters["full_attention"] = {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": float(
                    rope_full.parts[rope_full.data[0]].tolist()[0]
                ),
            }
        if rope_parameters:
            cfg["rope_parameters"] = rope_parameters
        cfg.pop("rope_theta", None)
        return parsed

    _mgu.load_gguf_checkpoint = _patched_load
    # transformers.configuration_utils imports the function by name at module
    # load, so patching the source module alone misses the path that
    # PretrainedConfig.get_config_dict actually invokes.
    from transformers import configuration_utils as _cu

    _cu.load_gguf_checkpoint = _patched_load

    # The transformers GGUF tokenizer converter doesn't propagate
    # ``tokenizer.ggml.add_bos_token`` from the GGUF metadata, so the
    # resulting fast tokenizer has ``add_bos_token=False`` even when the
    # underlying model requires a BOS prefix. For Gemma4 (which was trained
    # with a leading <bos>) that produces degenerate output — the model
    # collapses to repeating the last two prompt tokens. Patch
    # AutoTokenizer.from_pretrained to inject ``add_bos_token=True`` when
    # loading a gemma4 GGUF, and post-fix ``bos_token_id`` to whatever the
    # GGUF metadata reports (HF's converter sometimes leaves this at the
    # wrong vocab id — e.g. 203 instead of 2 for Gemma4).
    from transformers import AutoTokenizer

    _orig_from_pretrained = AutoTokenizer.from_pretrained

    def _patched_tokenizer_from_pretrained(
        pretrained_model_name_or_path, *args, **kwargs
    ):
        # vLLM passes the gguf basename as ``gguf_file`` alongside the
        # model path (which may be either the .gguf file itself or its
        # parent directory). Locate the actual file from that pair.
        gguf_path: Path | None = None
        base = Path(pretrained_model_name_or_path)
        gguf_file = kwargs.get("gguf_file")
        if gguf_file:
            parent = base.parent if base.is_file() else base
            candidate = parent / Path(gguf_file).name
            if candidate.is_file():
                gguf_path = candidate
        elif base.suffix == ".gguf" and base.is_file():
            gguf_path = base

        gguf_bos: int | None = None
        is_gemma4 = False
        if gguf_path is not None:
            try:
                reader = gguf.GGUFReader(gguf_path)
                arch_field = reader.fields.get("general.architecture")
                if arch_field is not None:
                    arch_bytes = bytes(arch_field.parts[arch_field.data[0]])
                    is_gemma4 = arch_bytes == b"gemma4"
                if is_gemma4:
                    bos_field = reader.fields.get("tokenizer.ggml.bos_token_id")
                    if bos_field is not None:
                        gguf_bos = int(
                            bos_field.parts[bos_field.data[0]].tolist()[0]
                        )
            except Exception as exc:
                logger.debug("gguf BOS probe failed for %s: %s", gguf_path, exc)

        if is_gemma4 and "add_bos_token" not in kwargs:
            kwargs["add_bos_token"] = True

        tokenizer = _orig_from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        if is_gemma4 and gguf_bos is not None:
            try:
                bos_str = tokenizer.convert_ids_to_tokens(gguf_bos)
                if bos_str:
                    tokenizer.bos_token = bos_str
                tokenizer.bos_token_id = gguf_bos
            except Exception as exc:
                logger.debug("failed to set BOS attrs on tokenizer: %s", exc)

            # The fast tokenizer's post_processor template was baked at
            # conversion time with the wrong BOS id (often 203 for Gemma4
            # variants), so even after fixing the bos_token_id attribute,
            # encode() prepends the wrong id. Replace the post_processor
            # to use the GGUF-declared BOS.
            try:
                from tokenizers import processors

                tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
                    single=f"{bos_str} $A",
                    pair=f"{bos_str} $A {bos_str} $B",
                    special_tokens=[(bos_str, gguf_bos)],
                )
            except Exception as exc:
                logger.debug(
                    "failed to override post_processor for BOS: %s", exc
                )
        return tokenizer

    AutoTokenizer.from_pretrained = _patched_tokenizer_from_pretrained


_register_gemma4_gguf_support()


@cache
def check_gguf_file(model: str | PathLike) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    try:
        with model.open("rb") as f:
            header = f.read(4)

        return header == b"GGUF"
    except Exception as e:
        logger.debug("Error reading file %s: %s", model, e)
        return False


@cache
def is_remote_gguf(model: str | Path) -> bool:
    """Check if the model is a remote GGUF model.

    Recognizes two forms:
    1. Standard: ``repo_id:quant_type`` where *quant_type* is a known
       GGML quantization type (e.g. ``Q4_K_M``).
    2. Non-standard: ``repo_id:quant_type`` where *quant_type* contains
       a known GGML type with extra prefixes (e.g. ``UD-Q4_K_XL``).
       A warning is logged and actual file existence is validated later
       during download.
    """
    pattern = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*:[A-Za-z0-9_+-]+$"
    model = str(model)
    if re.fullmatch(pattern, model):
        _, quant_type = model.rsplit(":", 1)
        if is_valid_gguf_quant_type(quant_type):
            return True
        if is_nonstandard_gguf_quant_type(quant_type):
            logger.warning(
                "Non-standard GGUF quant type '%s' detected.",
                quant_type,
            )
            return True
    return False


def is_nonstandard_gguf_quant_type(quant_type: str) -> bool:
    """Check if a non-standard quant type contains a known GGML type.

    Splits the quant type by the last ``-`` and checks whether the
    trailing part is a standard GGML type.  For example::

        UD-Q4_K_XL      → rsplit → ["UD", "Q4_K_XL"]      → Q4_K_XL valid ✓
        UD-IQ4_NL       → rsplit → ["UD", "IQ4_NL"]       → IQ4_NL  valid ✓
        Custom-UD-Q4_K  → rsplit → ["Custom-UD", "Q4_K"]  → Q4_K    valid ✓
        RANDOM          → no "-" → False
    """
    if "-" not in quant_type:
        return False
    _, remainder = quant_type.rsplit("-", 1)
    return is_valid_gguf_quant_type(remainder)


# Common suffixes used in GGUF file naming conventions
# e.g., Q4_K_M, Q3_K_S, Q5_K_L, Q2_K_XL
_GGUF_QUANT_SUFFIXES = ("_M", "_S", "_L", "_XL", "_XS", "_XXS")


def is_valid_gguf_quant_type(gguf_quant_type: str) -> bool:
    """Check if the quant type is a valid GGUF quant type.

    Supports both exact GGML quant types (e.g., Q4_K, IQ1_S) and
    extended naming conventions (e.g., Q4_K_M, Q3_K_S, Q5_K_L).
    """
    # Check for exact match first
    if getattr(GGMLQuantizationType, gguf_quant_type, None) is not None:
        return True

    # Check for extended naming conventions (e.g., Q4_K_M -> Q4_K)
    for suffix in _GGUF_QUANT_SUFFIXES:
        if gguf_quant_type.endswith(suffix):
            base_type = gguf_quant_type[: -len(suffix)]
            if getattr(GGMLQuantizationType, base_type, None) is not None:
                return True

    return False


def split_remote_gguf(model: str | Path) -> tuple[str, str]:
    """Split the model into repo_id and quant type."""
    model = str(model)
    if is_remote_gguf(model):
        parts = model.rsplit(":", 1)
        return (parts[0], parts[1])
    raise ValueError(
        f"Wrong GGUF model or invalid GGUF quant type: {model}.\n"
        "- It should be in repo_id:quant_type format.\n"
        f"- Valid base quant types: {GGMLQuantizationType._member_names_}\n"
        f"- Extended suffixes also supported: {_GGUF_QUANT_SUFFIXES}\n"
        "- Non-standard GGUF quant types also supported: "
        "dash-separated prefixes (e.g. UD-Q4_K_XL, Custom-Q8_0)",
    )


def is_gguf(model: str | Path) -> bool:
    """Check if the model is a GGUF model.

    Args:
        model: Model name, path, or Path object to check.

    Returns:
        True if the model is a GGUF model, False otherwise.
    """
    model = str(model)

    # Check if it's a local GGUF file
    if check_gguf_file(model):
        return True

    # Check if it's a remote GGUF model (repo_id:quant_type format)
    return is_remote_gguf(model)


def detect_gguf_multimodal(model: str) -> Path | None:
    """Check if GGUF model has multimodal projector file.

    Args:
        model: Model path string

    Returns:
        Path to mmproj file if found, None otherwise
    """
    if not model.endswith(".gguf"):
        return None

    try:
        model_path = Path(model)
        if not model_path.is_file():
            return None

        model_dir = model_path.parent
        mmproj_patterns = ["mmproj.gguf", "mmproj-*.gguf", "*mmproj*.gguf"]
        for pattern in mmproj_patterns:
            mmproj_files = list(model_dir.glob(pattern))
            if mmproj_files:
                return mmproj_files[0]
        return None
    except Exception:
        return None


def extract_vision_config_from_gguf(mmproj_path: str) -> "SiglipVisionConfig | None":
    """Extract vision config parameters from mmproj.gguf metadata.

    Reads vision encoder configuration from GGUF metadata fields using
    standardized GGUF constants. Automatically detects the projector type
    (e.g., gemma3, llama4) and applies model-specific parameters accordingly.

    The function extracts standard CLIP vision parameters from GGUF metadata
    and applies projector-type-specific customizations. For unknown projector
    types, it uses safe defaults from SiglipVisionConfig.

    Args:
        mmproj_path: Path to mmproj.gguf file (str or Path)

    Returns:
        SiglipVisionConfig if extraction succeeds, None if any required
        field is missing from the GGUF metadata

    Raises:
        Exception: Exceptions from GGUF reading (file not found, corrupted
            file, etc.) propagate directly from gguf.GGUFReader
    """
    reader = gguf.GGUFReader(str(mmproj_path))

    # Detect projector type to apply model-specific parameters
    projector_type = None
    projector_type_field = reader.get_field(Keys.Clip.PROJECTOR_TYPE)
    if projector_type_field:
        try:
            projector_type = bytes(projector_type_field.parts[-1]).decode("utf-8")
        except (AttributeError, UnicodeDecodeError) as e:
            logger.warning("Failed to decode projector type from GGUF: %s", e)

    # Map GGUF field constants to SiglipVisionConfig parameters.
    # Uses official GGUF constants from gguf-py for standardization.
    # Format: {gguf_constant: (param_name, dtype)}
    VISION_CONFIG_FIELDS = {
        Keys.ClipVision.EMBEDDING_LENGTH: ("hidden_size", int),
        Keys.ClipVision.FEED_FORWARD_LENGTH: ("intermediate_size", int),
        Keys.ClipVision.BLOCK_COUNT: ("num_hidden_layers", int),
        Keys.ClipVision.Attention.HEAD_COUNT: ("num_attention_heads", int),
        Keys.ClipVision.IMAGE_SIZE: ("image_size", int),
        Keys.ClipVision.PATCH_SIZE: ("patch_size", int),
        Keys.ClipVision.Attention.LAYERNORM_EPS: ("layer_norm_eps", float),
    }

    # Extract and validate all required fields
    config_params = {}
    for gguf_key, (param_name, dtype) in VISION_CONFIG_FIELDS.items():
        field = reader.get_field(gguf_key)
        if field is None:
            logger.warning(
                "Missing required vision config field '%s' in mmproj.gguf",
                gguf_key,
            )
            return None
        # Extract scalar value from GGUF field and convert to target type
        config_params[param_name] = dtype(field.parts[-1])

    # Apply model-specific parameters based on projector type
    if projector_type == VisionProjectorType.GEMMA3:
        # Gemma3 doesn't use the vision pooling head (multihead attention)
        # This is a vLLM-specific parameter used in SiglipVisionTransformer
        config_params["vision_use_head"] = False
        logger.info("Detected Gemma3 projector, disabling vision pooling head")
    # Add other projector-type-specific customizations here as needed
    # elif projector_type == VisionProjectorType.LLAMA4:
    #     config_params["vision_use_head"] = ...

    # Create config with extracted parameters
    # Note: num_channels and attention_dropout use SiglipVisionConfig defaults
    # (3 and 0.0 respectively) which are correct for all models
    config = SiglipVisionConfig(**config_params)

    if projector_type:
        logger.info(
            "Extracted vision config from mmproj.gguf (projector_type: %s)",
            projector_type,
        )
    else:
        logger.info("Extracted vision config from mmproj.gguf metadata")

    return config


def maybe_patch_hf_config_from_gguf(
    model: str,
    hf_config: PretrainedConfig,
) -> PretrainedConfig:
    """Patch HF config for GGUF models.

    Applies GGUF-specific patches to HuggingFace config:
    1. For multimodal models: patches architecture and vision config
    2. For all GGUF models: overrides vocab_size from embedding tensor

    This ensures compatibility with GGUF models that have extended
    vocabularies (e.g., Unsloth) where the GGUF file contains more
    tokens than the HuggingFace tokenizer config specifies.

    Args:
        model: Model path string
        hf_config: HuggingFace config to patch in-place

    Returns:
        Updated HuggingFace config
    """
    # Patch multimodal config if mmproj.gguf exists
    mmproj_path = detect_gguf_multimodal(model)
    if mmproj_path is not None:
        vision_config = extract_vision_config_from_gguf(str(mmproj_path))

        # Create HF config for Gemma3 multimodal
        text_config = hf_config.get_text_config()
        is_gemma3 = hf_config.model_type in ("gemma3", "gemma3_text")
        if vision_config is not None and is_gemma3:
            new_hf_config = Gemma3Config(
                text_config=text_config,
                vision_config=vision_config,
                architectures=["Gemma3ForConditionalGeneration"],
            )
            hf_config = new_hf_config

    return hf_config


def get_gguf_file_path_from_hf(
    repo_id: str | Path,
    quant_type: str,
    revision: str | None = None,
) -> str:
    """Get the GGUF file path from HuggingFace Hub based on repo_id and quant_type.

    Args:
        repo_id: The HuggingFace repository ID (e.g., "Qwen/Qwen3-0.6B")
        quant_type: The quantization type (e.g., "Q4_K_M", "F16")
        revision: Optional revision/branch name

    Returns:
        The path to the GGUF file on HuggingFace Hub (e.g., "filename.gguf"),
    """
    repo_id = str(repo_id)
    gguf_patterns = [
        f"*-{quant_type}.gguf",
        f"*-{quant_type}-*.gguf",
        f"*/*-{quant_type}.gguf",
        f"*/*-{quant_type}-*.gguf",
    ]
    matching_files = list_filtered_repo_files(
        repo_id,
        allow_patterns=gguf_patterns,
        revision=revision,
    )

    if len(matching_files) == 0:
        raise ValueError(
            "Could not find GGUF file for repo %s with quantization %s.",
            repo_id,
            quant_type,
        )

    # Sort to ensure consistent ordering (prefer non-sharded files)
    matching_files.sort(key=lambda x: (x.count("-"), x))
    gguf_filename = matching_files[0]
    return gguf_filename
