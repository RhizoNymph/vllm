"""Phase 2: capture parity — this fork's residuals vs Megatron ground truth.

For each Megatron reference capture (fit_jlens.py --jlens-capture-residuals
dump), send the exact same raw text through /v1/completions (no chat
template) with a filesystem capture request at the lens band, read the .bin
files back with the fork's own reader, and compare per-position cosine
similarity at each layer.

Validates in one shot: hook-point equivalence (post_block == Megatron block
output), tokenizer agreement, and — since the server runs fp8 — quantization
fidelity at the hook points.

Usage:
    python phase2_parity.py \
        --refs /mnt/data/artifacts/jlens/glm52_capture_demo \
        --capture-root /mnt/data/artifacts/jlens/vllm_capture \
        --layers 30 40 50
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import urllib.request

import numpy as np
import torch

from vllm.v1.capture.consumers.filesystem.reader import read_per_file


def _to_float(array: np.ndarray) -> torch.Tensor:
    """bf16 captures come back as raw uint16 bit patterns (numpy has no
    bfloat16 dtype); reinterpret before any arithmetic."""
    t = torch.from_numpy(array.copy())
    if array.dtype == np.uint16:
        t = t.view(torch.bfloat16)
    return t.float()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refs", default="/mnt/data/artifacts/jlens/glm52_capture_demo")
    p.add_argument("--capture-root", default="/mnt/data/artifacts/jlens/vllm_capture")
    p.add_argument("--layers", type=int, nargs="+", default=[30, 40, 50])
    p.add_argument("--run-info", default="/mnt/data/artifacts/jlens/serve/current.json")
    p.add_argument("--tag", default="parity")
    args = p.parse_args()

    info = json.load(open(args.run_info))
    base = f"http://{info['host']}:{info['port']}"
    refs = sorted(glob.glob(os.path.join(args.refs, "capture_*.pt")))
    print(f"server {base} ({info['quantization'] or 'bf16'}); {len(refs)} reference prompts")

    worst = 1.0
    for path in refs:
        ref = torch.load(path, map_location="cpu", weights_only=True)
        rid = os.path.splitext(os.path.basename(path))[0]  # capture_0000
        payload = {
            "model": "glm-5.2",
            "prompt": ref["text"],
            "max_tokens": 1,
            "temperature": 0,
            "capture": {
                "filesystem": {
                    "request_id": rid,
                    "tag": args.tag,
                    "hooks": {"post_block": args.layers},
                    "positions": "all_prompt",
                    "layout": "per_file",
                }
            },
            "capture_wait": True,
        }
        req = urllib.request.Request(
            f"{base}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            json.load(r)

        row = [rid]
        for layer in args.layers:
            bin_path = os.path.join(
                args.capture_root, args.tag, rid, f"{layer}_post_block.bin"
            )
            entry = read_per_file(bin_path)
            got = _to_float(entry.array)  # [rows, 6144]
            want = ref["layers"][layer].float()  # [seq, 6144]
            n = min(got.shape[0], want.shape[0])
            if got.shape[0] != want.shape[0]:
                row.append(f"L{layer}: ROWS {got.shape[0]} vs {want.shape[0]}!")
            cos = torch.nn.functional.cosine_similarity(got[:n], want[:n], dim=-1)
            worst = min(worst, float(cos.min()))
            row.append(f"L{layer}: min {cos.min():.4f} mean {cos.mean():.4f}")
        print("  " + "  ".join(row))

    print(f"\nworst per-position cosine: {worst:.4f} "
          f"({'PASS >0.99' if worst > 0.99 else 'BELOW 0.99 — check fp8/alignment'})")


if __name__ == "__main__":
    main()
