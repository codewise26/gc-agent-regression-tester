"""CLI entry point for the GC Agent Regression Tester.

Parses command-line arguments, loads configuration, runs the test suite,
prints progress and results to the console, and exits with a non-zero
code if regressions are detected.
"""

import argparse
import asyncio
import sys
import threading
from pathlib import Path

from .app_config import (
    load_app_config,
    validate_platform_config,
    validate_required_config,
)
from .cli_commands import (
    disconnect_all_conversations,
    disconnect_conversation,
    list_conversations,
    load_config_for_management,
)
from .config_loader import load_test_suite
from .judge_llm import JudgeLLMClient, JudgeLLMError
from .models import AppConfig, ProgressEvent, ProgressEventType
from .orchestrator import TestOrchestrator
from .platform_api_client import PlatformApiError
from .progress import ProgressEmitter


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "test_suite",
        help="Path to test suite file (JSON or YAML)",
    )
    parser.add_argument(
        "--region",
        help="Genesys Cloud region override",
    )
    parser.add_argument(
        "--deployment-id",
        help="Genesys Cloud deployment ID override",
    )
    parser.add_argument(
        "--ollama-url",
        help="Ollama base URL override",
    )
    parser.add_argument(
        "--ollama-model",
        help="Ollama model name override",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        help="Default number of attempts per scenario override",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="Maximum conversation turns override",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Response timeout in seconds override",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Success threshold (0.0-1.0) override",
    )


def _parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    # Backward compatibility: `python -m src.cli suite.yaml` -> run suite.yaml
    if raw_argv and not raw_argv[0].startswith("-") and raw_argv[0] not in {
        "run",
        "disconnect",
        "conversations",
    }:
        suffix = Path(raw_argv[0]).suffix.lower()
        if suffix in {".yaml", ".yml", ".json"}:
            raw_argv = ["run", *raw_argv]

    parser = argparse.ArgumentParser(
        description="GC Agent Regression Tester — LLM-as-judge testing for Genesys Cloud agents"
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Run a regression test suite",
    )
    _add_run_arguments(run_parser)

    disconnect_parser = subparsers.add_parser(
        "disconnect",
        help="Disconnect tracked Genesys conversations via Platform API",
    )
    disconnect_group = disconnect_parser.add_mutually_exclusive_group(required=True)
    disconnect_group.add_argument(
        "--id",
        dest="conversation_id",
        help="Disconnect a single conversation by ID",
    )
    disconnect_group.add_argument(
        "--all",
        action="store_true",
        help="Disconnect all tracked conversations",
    )

    list_parser = subparsers.add_parser(
        "conversations",
        help="Manage tracked conversations",
    )
    list_subparsers = list_parser.add_subparsers(dest="conversations_command")
    list_subparsers.add_parser("list", help="List tracked active conversations")

    return parser.parse_args(raw_argv)


def _merge_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Merge CLI argument overrides into the base config."""
    data = config.model_dump()

    if getattr(args, "region", None) is not None:
        data["gc_region"] = args.region
    if getattr(args, "deployment_id", None) is not None:
        data["gc_deployment_id"] = args.deployment_id
    if getattr(args, "ollama_url", None) is not None:
        data["ollama_base_url"] = args.ollama_url
    if getattr(args, "ollama_model", None) is not None:
        data["ollama_model"] = args.ollama_model
    if getattr(args, "attempts", None) is not None:
        data["default_attempts"] = args.attempts
    if getattr(args, "max_turns", None) is not None:
        data["max_turns"] = args.max_turns
    if getattr(args, "timeout", None) is not None:
        data["response_timeout"] = args.timeout
    if getattr(args, "threshold", None) is not None:
        data["success_threshold"] = args.threshold

    return AppConfig(**data)


def _progress_printer(progress_queue, stop_event: threading.Event) -> None:
    """Print progress events from the queue to the console."""
    while not stop_event.is_set() or not progress_queue.empty():
        try:
            event: ProgressEvent = progress_queue.get(timeout=0.5)
            _print_progress_event(event)
        except Exception:
            continue


def _print_progress_event(event: ProgressEvent) -> None:
    """Format and print a single progress event to the console."""
    prefix = {
        ProgressEventType.SUITE_STARTED: "🚀",
        ProgressEventType.SCENARIO_STARTED: "📋",
        ProgressEventType.ATTEMPT_STARTED: "▶",
        ProgressEventType.DEBUG_STEP: "··",
        ProgressEventType.ATTEMPT_COMPLETED: "  ✓" if event.success else "  ✗",
        ProgressEventType.SCENARIO_COMPLETED: "📊",
        ProgressEventType.SUITE_COMPLETED: "🏁",
    }.get(event.event_type, "•")

    line = f"{prefix} {event.message}"
    if event.detail:
        line += f" — {event.detail}"
    print(line)


def _print_report(report) -> None:
    """Print a formatted test report summary to the console."""
    print("\n" + "=" * 60)
    print(f"TEST REPORT: {report.suite_name}")
    print("=" * 60)
    print(f"Duration: {report.duration_seconds:.1f}s")
    print(f"Overall: {report.overall_successes}/{report.overall_attempts} "
          f"({report.overall_success_rate:.0%} success rate)")
    print(f"Threshold: {report.regression_threshold:.0%}")
    print("-" * 60)

    for result in report.scenario_results:
        status = "⚠️  REGRESSION" if result.is_regression else "✅ PASS"
        print(f"  {result.scenario_name}: "
              f"{result.successes}/{result.attempts} "
              f"({result.success_rate:.0%}) — {status}")

    print("-" * 60)
    if report.has_regressions:
        print("❌ REGRESSIONS DETECTED")
    else:
        print("✅ ALL SCENARIOS PASSED")
    print("=" * 60)


def _run_suite_command(args: argparse.Namespace) -> int:
    config = load_app_config()
    config = _merge_cli_overrides(config, args)

    try:
        suite = load_test_suite(args.test_suite)
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error loading test suite: {e}", file=sys.stderr)
        return 1

    missing = validate_required_config(config)
    if missing:
        print(
            f"Error: Missing required configuration: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    platform_missing = validate_platform_config(config)
    if platform_missing:
        print(
            "Error: Missing Platform API configuration required for conversation cleanup: "
            f"{', '.join(platform_missing)}",
            file=sys.stderr,
        )
        return 1

    judge = JudgeLLMClient(
        base_url=config.ollama_base_url,
        model=config.ollama_model or "",
        timeout=config.response_timeout,
    )
    try:
        judge.verify_connection()
    except JudgeLLMError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    emitter = ProgressEmitter()
    progress_queue = emitter.subscribe()
    stop_event = threading.Event()
    printer_thread = threading.Thread(
        target=_progress_printer,
        args=(progress_queue, stop_event),
        daemon=True,
    )
    printer_thread.start()

    orchestrator = TestOrchestrator(config=config, progress_emitter=emitter)
    report = asyncio.run(orchestrator.run_suite(suite))

    stop_event.set()
    printer_thread.join(timeout=5)

    _print_report(report)
    return 1 if report.has_regressions else 0


def _disconnect_command(args: argparse.Namespace) -> int:
    config = load_config_for_management()

    try:
        if args.all:
            success_count, errors = disconnect_all_conversations(config)
            print(f"Disconnected {success_count} conversation(s).")
            for error in errors:
                print(f"Warning: {error}", file=sys.stderr)
            return 1 if errors else 0

        disconnect_conversation(config, args.conversation_id)
        print(f"Disconnected conversation {args.conversation_id}.")
        return 0
    except PlatformApiError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _conversations_command(args: argparse.Namespace) -> int:
    if args.conversations_command != "list":
        print("Error: Unknown conversations command.", file=sys.stderr)
        return 1

    config = load_config_for_management()
    entries = list_conversations(config)

    if not entries:
        print("No active conversations tracked.")
        return 0

    for entry in entries:
        print(
            f"{entry.conversation_id}\t"
            f"scenario={entry.scenario_name}\t"
            f"attempt={entry.attempt_number}\t"
            f"registered_at={entry.registered_at}"
        )
    return 0


def main(argv=None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)

    if args.command in (None, "run"):
        if args.command is None:
            print("Error: No command specified.", file=sys.stderr)
            sys.exit(1)
        sys.exit(_run_suite_command(args))

    if args.command == "disconnect":
        sys.exit(_disconnect_command(args))

    if args.command == "conversations":
        sys.exit(_conversations_command(args))

    print(f"Error: Unknown command: {args.command}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
