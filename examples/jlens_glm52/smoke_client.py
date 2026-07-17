"""Phase-0 smoke for the GLM-5.2 jlens server: plain completion, steering
surface check, and (optionally) a steered completion from a packed spec.

Reads the server address from the RUN_INFO json written by
sbatch-serve-glm52.sh.

Usage:
    python smoke_client.py                       # baseline + endpoint checks
    python smoke_client.py --steer dog_s4.json   # + steered comparison
"""

from __future__ import annotations

import argparse
import json

import urllib.request


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def _chat(base: str, prompt: str, extra_body: dict | None = None) -> str:
    payload = {
        "model": "glm-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 96,
        "temperature": 0,
        "seed": 0,
        **(extra_body or {}),
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-info", default="/mnt/data/artifacts/jlens/serve/current.json")
    p.add_argument("--prompt", default="Tell me about your weekend plans.")
    p.add_argument("--steer", help="packed steering spec json (pack_steering.py)")
    args = p.parse_args()

    info = json.load(open(args.run_info))
    base = f"http://{info['host']}:{info['port']}"
    print(f"server: {base} (job {info['job']}, {info['quantization'] or 'bf16'})")

    layers = _get(f"{base}/v1/steering/layers")
    print(f"steering surface: {json.dumps(layers)[:160]}...")

    baseline = _chat(base, args.prompt)
    print(f"\n--- baseline ---\n{baseline}")

    if args.steer:
        spec = json.load(open(args.steer))
        # raw HTTP: what openai-python calls extra_body is just top-level fields
        steered = _chat(base, args.prompt, {"decode_steering_vectors": spec})
        print(f"\n--- steered ({args.steer}) ---\n{steered}")
        print(f"\nchanged: {steered != baseline}")


if __name__ == "__main__":
    main()
