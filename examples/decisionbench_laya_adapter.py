"""Standalone DecisionBench (`Hanno-Labs/decision-bench`) runner & server bridge for `laya-as-judge`.

Usage (reproducible by both contributors and Hanno Labs maintainers):
  1. Verify native `/v1/systemone` contract compatibility (in-process emulator or live server):
       python examples/decisionbench_laya_adapter.py --verify-only

  2. Start `laya-judge serve` (e.g., `rtx-16gb-glide` or `edge-6gb`) and invoke
     `Hanno-Labs/decision-bench`'s official `run-system-one-http` harness:
       python examples/decisionbench_laya_adapter.py \
         --decision-bench-dir /path/to/decision-bench \
         --output-dir ./results/laya-rtx-16gb-glide \
         --base-url http://127.0.0.1:8000 \
         --profile rtx-16gb-glide \
         --smoke

  3. Print a pre-filled `Hanno-Labs/decision-bench` `[Evaluation]` GitHub issue:
       python examples/decisionbench_laya_adapter.py --print-issue --profile rtx-16gb-glide
"""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import threading
import time
from typing import Optional

from laya_as_judge.decision_bench import (
    build_decision_bench_command,
    format_evaluation_request_issue,
    run_decision_bench_subprocess,
    verify_systemone_contract,
)


def _start_embedded_server(profile: str, tier1_url: Optional[str], tier2_url: Optional[str]):
    import uvicorn
    from laya_as_judge import create_app

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])

    app = create_app(profile=profile, tier1_url=tier1_url, tier2_url=tier2_url)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.35)
    return server, thread, f"http://127.0.0.1:{port}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or verify laya-as-judge against Hanno-Labs/decision-bench.")
    parser.add_argument("--base-url", default=None, help="Existing `laya-judge serve` URL (if omitted, starts embedded server).")
    parser.add_argument("--profile", default="emulator", help="Hardware profile when starting embedded server.")
    parser.add_argument("--tier1-url", default=None, help="Optional Tier-1 llama-server URL.")
    parser.add_argument("--tier2-url", default=None, help="Optional Tier-2 llama-server URL.")
    parser.add_argument("--decision-bench-dir", default=None, help="Path to local `Hanno-Labs/decision-bench` checkout.")
    parser.add_argument("--output-dir", default="./decisionbench_results", help="Output directory for DecisionBench artifacts.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Pass --smoke to decision-bench.")
    parser.add_argument("--verify-only", action="store_true", help="Only run the 3-primitive /v1/systemone contract check.")
    parser.add_argument("--print-issue", action="store_true", help="Print pre-filled Hanno-Labs/decision-bench evaluation issue.")
    args = parser.parse_args()

    if args.print_issue:
        print(format_evaluation_request_issue(profile=args.profile))
        return

    server = thread = None
    base_url = args.base_url
    try:
        if not base_url:
            server, thread, base_url = _start_embedded_server(args.profile, args.tier1_url, args.tier2_url)

        check = verify_systemone_contract(base_url, model=f"laya-as-judge/{args.profile}")
        print(json.dumps(check, indent=2))
        if not check["passed"]:
            raise SystemExit(1)

        if args.verify_only or not args.decision_bench_dir:
            if not args.verify_only:
                cmd = build_decision_bench_command(
                    decision_bench_dir="/path/to/decision-bench",
                    output_dir=args.output_dir,
                    base_url=base_url,
                    profile=args.profile,
                    smoke=args.smoke,
                )
                print("\nNext step — run official DecisionBench harness:\n ", shlex.join(cmd))
            return

        run_decision_bench_subprocess(
            decision_bench_dir=args.decision_bench_dir,
            output_dir=args.output_dir,
            base_url=base_url,
            profile=args.profile,
            concurrency=args.concurrency,
            smoke=args.smoke,
        )
    finally:
        if server is not None:
            server.should_exit = True
            if thread is not None:
                thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
