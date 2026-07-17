"""Pack Jacobian-lens steering directions into this fork's wire format.

Input: a .pt from MegaFire's ``examples/jlens/apply_jlens.py --export-steering``
(``{"directions": {layer: [hidden] fp32}, "word", "token_id", "strength",
"calibrated"}``). Output: ``SteeringVectorSpecPacked`` JSON —
``{hook: {dtype, shape, layer_indices, data: base64, scales}}`` — usable as
``extra_body.steering_vectors`` / ``decode_steering_vectors`` on a chat
request, or wrapped for ``--steering-modules`` / ``/v1/steering/modules``.

Usage:
    python pack_steering.py --pt /mnt/data/artifacts/jlens/steering_dog_demo.pt \
        --strength 4 --hook post_block --out dog_s4.json
"""

from __future__ import annotations

import argparse
import base64
import json

import numpy as np
import torch


def pack(directions: dict[int, torch.Tensor], *, hook: str, scale: float) -> dict:
    layers = sorted(directions)
    mat = np.stack([directions[l].float().numpy() for l in layers]).astype("<f4")
    return {
        hook: {
            "dtype": "float32",
            "shape": [len(layers), mat.shape[1]],
            "layer_indices": layers,
            "data": base64.b64encode(np.ascontiguousarray(mat).tobytes()).decode(),
            "scales": [scale] * len(layers),
        }
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pt", required=True, help="apply_jlens.py --export-steering output")
    p.add_argument("--hook", default="post_block",
                   choices=["pre_attn", "post_attn", "post_block"])
    p.add_argument("--strength", type=float, default=None,
                   help="Desired strength; vectors are rescaled by "
                   "strength / baked-in strength. Default: keep as exported.")
    p.add_argument("--layers", type=int, nargs="*",
                   help="Subset of exported layers (default: all).")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    bundle = torch.load(args.pt, map_location="cpu", weights_only=True)
    directions = bundle["directions"]
    if args.layers:
        missing = set(args.layers) - set(directions)
        if missing:
            raise SystemExit(f"layers {sorted(missing)} not in {args.pt} "
                             f"(has {sorted(directions)})")
        directions = {l: directions[l] for l in args.layers}
    scale = 1.0
    if args.strength is not None:
        baked = float(bundle.get("strength", 1.0))
        scale = args.strength / baked
    spec = pack(directions, hook=args.hook, scale=scale)

    with open(args.out, "w") as f:
        json.dump(spec, f)
    word = bundle.get("word", "?")
    print(f"packed {len(directions)} layers for {word!r} -> {args.out} "
          f"(hook={args.hook}, scale={scale:g}, "
          f"calibrated={bundle.get('calibrated')})")


if __name__ == "__main__":
    main()
