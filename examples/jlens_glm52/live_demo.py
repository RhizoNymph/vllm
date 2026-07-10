"""Phase 3/4 demo: live Jacobian-lens readout during generation (optionally
while steering the same request).

Sends a chat completion with the jlens capture consumer armed; the consumer
streams per-token band readouts to JSONL as the model generates. Afterwards
this script aligns the readout rows with the generated tokens and renders a
table: generated token | lens top-1 at each band layer.

Usage:
    python live_demo.py --prompt "Fact: The currency used in the country \
shaped like a boot is" --raw
    python live_demo.py --steer /mnt/data/artifacts/jlens/serve/dog40_s0.8.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import urllib.request


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-info", default="/mnt/data/artifacts/jlens/serve/current.json")
    p.add_argument("--readout-dir", default="/mnt/data/artifacts/jlens/readout")
    p.add_argument("--prompt", default="Tell me about your weekend plans.")
    p.add_argument("--raw", action="store_true",
                   help="use /v1/completions (no chat template)")
    p.add_argument("--steer", help="packed steering spec json (pack_steering.py)")
    p.add_argument("--layers", type=int, nargs="+", default=[30, 40, 50])
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--out", help="append a markdown transcript here")
    args = p.parse_args()

    info = json.load(open(args.run_info))
    base = f"http://{info['host']}:{info['port']}"

    payload: dict = {
        "model": "glm-5.2",
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": 0,
        "capture": {
            "jlens": {
                "hooks": {"post_block": args.layers},
                "positions": "all_generated",
            }
        },
        "capture_wait": True,
    }
    if args.raw:
        endpoint, key = "/v1/completions", "text"
        payload["prompt"] = args.prompt
    else:
        endpoint, key = "/v1/chat/completions", None
        payload["messages"] = [{"role": "user", "content": args.prompt}]
    if args.steer:
        payload["decode_steering_vectors"] = json.load(open(args.steer))

    before = time.time()
    req = urllib.request.Request(
        f"{base}{endpoint}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    choice = resp["choices"][0]
    text = choice[key] if key else choice["message"]["content"]

    # Find this request's readout file: newest jsonl written since `before`.
    time.sleep(1.0)
    candidates = [
        f for f in glob.glob(os.path.join(args.readout_dir, "*.jsonl"))
        if os.path.getmtime(f) >= before - 1
    ]
    if not candidates:
        raise SystemExit(f"no readout file appeared under {args.readout_dir}")
    readout_path = max(candidates, key=os.path.getmtime)

    # rows: pos -> {layer: [top tokens]}
    per_pos: dict[int, dict[int, list[str]]] = {}
    for line in open(readout_path, encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("event"):
            continue
        for pos, top in zip(rec["pos"], rec["top"]):
            per_pos.setdefault(pos, {})[rec["layer"]] = top

    # Align: generated token at logical position p predicts token p+1; the
    # readout at p is "what the model is disposed to say" while emitting it.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/mnt/data/artifacts/GLM-5.2",
                                        trust_remote_code=True)
    gen_ids = tok(text, add_special_tokens=False).input_ids
    n_prompt = resp["usage"]["prompt_tokens"]
    positions = sorted(per_pos)

    lines = [
        f"## prompt: {args.prompt!r}" + (f"  |  steer: {args.steer}" if args.steer else ""),
        "", f"completion ({resp['usage']['completion_tokens']} tokens):",
        "", "```", text, "```", "",
        "| gen token | " + " | ".join(f"L{l} top-1" for l in args.layers) + " |",
        "|---|" + "---|" * len(args.layers),
    ]
    for i, pos in enumerate(positions):
        tok_str = tok.decode([gen_ids[i]]) if i < len(gen_ids) else "?"
        row = [repr(tok_str)]
        for layer in args.layers:
            top = per_pos[pos].get(layer)
            row.append(repr(top[0]) if top else "-")
        lines.append("| " + " | ".join(row) + " |")
    report = "\n".join(lines)
    print(report)
    print(f"\n(readout stream: {readout_path}; {len(positions)} positions, "
          f"prompt had {n_prompt} tokens)")
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(report + "\n\n")


if __name__ == "__main__":
    main()
