"""Explicit, side-effect-scoped command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from .core import init_case, ingest_pdf, inspect_hub_model, inspect_local_model
from .startlux import analyze, probe


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paperrev", description="Evidence-first scientific paper reconstruction")
    sub = p.add_subparsers(dest="command", required=True)

    x = sub.add_parser("init", help="Freeze a paper case and initialize unverified claim/unknown ledgers")
    x.add_argument("--case", required=True)
    x.add_argument("--out", required=True)

    x = sub.add_parser("ingest-pdf", help="Extract source-bound page text (optional PyMuPDF dependency)")
    x.add_argument("--pdf", required=True)
    x.add_argument("--out", required=True)

    x = sub.add_parser("inspect-local", help="Read small model metadata files, never tensors")
    x.add_argument("--model-dir", required=True)
    x.add_argument("--out", required=True)

    x = sub.add_parser("inspect-hub", help="Pin HF Hub model revision and read configs, not weight bytes")
    x.add_argument("--repo", required=True)
    x.add_argument("--revision", default="main")
    x.add_argument("--out", required=True)

    x = sub.add_parser("probe-startlux", help="Record real always-think predictions from the official API")
    x.add_argument("--dataset", required=True)
    x.add_argument("--base-url", default="http://127.0.0.1:8090")
    x.add_argument("--out", required=True)
    x.add_argument("--execute", action="store_true", help="explicitly authorize HTTP inference calls")
    x.add_argument("--allow-remote", action="store_true", help="allow external HTTPS endpoint")
    x.add_argument("--limit", type=int, default=10)
    x.add_argument("--budget", type=int, default=256)
    x.add_argument("--mix", type=float, default=0.75)
    x.add_argument("--timeout", type=int, default=600)

    x = sub.add_parser("analyze-startlux", help="Offline confidence gate audit on frozen always-think captures")
    x.add_argument("--records", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--gates", default="0.6,0.9")
    x.add_argument("--mix", type=float, default=0.75)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = init_case(args.case, args.out)
        elif args.command == "ingest-pdf":
            result = ingest_pdf(args.pdf, args.out)
        elif args.command == "inspect-local":
            result = inspect_local_model(args.model_dir, args.out)
        elif args.command == "inspect-hub":
            result = inspect_hub_model(args.repo, args.revision, args.out)
        elif args.command == "probe-startlux":
            result = probe(
                args.dataset, args.out, args.base_url, execute=args.execute,
                allow_remote=args.allow_remote, max_requests=args.limit,
                think_budget=args.budget, mix=args.mix, timeout=args.timeout
            )
        elif args.command == "analyze-startlux":
            result = analyze(args.records, args.out, [float(x) for x in args.gates.split(",")], args.mix)
        else:
            parser.error("unrecognized command")
            return 2
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"paperrev: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") != "partial" else 2


if __name__ == "__main__":
    raise SystemExit(main())
