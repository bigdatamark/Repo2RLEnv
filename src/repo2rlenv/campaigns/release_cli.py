# [MODULE DOC] CLI adapter for the 'release' subcommand (stage | verify | publish) over
# campaigns.release: builds immutable Harbor dataset releases from an explicit plan and publishes
# them to the HF Hub.
"""Prepare and publish explicit, immutable selections of Harbor tasks."""

from __future__ import annotations

from pathlib import Path

from repo2rlenv.campaigns.release import (
    ReleasePlan,
    publish_release,
    recover_empty_upload,
    stage_release,
    verify_release,
)
from repo2rlenv.ui import console


def cmd_release(args):
    # [FUNCTION DOC] Handler for 'release'. stage: parses --plan JSON into ReleasePlan, runs
    # stage_release into --out, reports repo_id, task count and directory. verify: verify_release on
    # the directory, reports repo_id, file count, verified=True. publish: resolves an HF token
    # (resolve_hf_token with default AuthSpec; ValueError if none), then with --recover-empty calls
    # recover_empty_upload (default batch size 250), else publish_release with --receipt,
    # --collection and --batch-size; reports the receipt record.
    # Prints JSON (--json) or a 'Harbor dataset release' panel. Returns 0. Propagates errors from
    # the release functions. Side effects: filesystem, network, console output.
    if args.release_action == "stage":
        plan = ReleasePlan.model_validate_json(args.plan.read_text())
        result = stage_release(plan, args.out)
        data = {
            "repo_id": result["repo_id"],
            "tasks": result["task_count"],
            "directory": str(args.out),
        }
    elif args.release_action == "verify":
        result = verify_release(args.directory)
        data = {"repo_id": result["repo_id"], "files": len(result["files"]), "verified": True}
    else:
        from huggingface_hub import HfApi

        from repo2rlenv.auth import resolve_hf_token
        from repo2rlenv.spec.input import AuthSpec

        token = resolve_hf_token(AuthSpec())
        if not token:
            raise ValueError("Publishing requires HF_TOKEN or a configured Hub login")
        if args.recover_empty:
            data = recover_empty_upload(
                args.directory, api=HfApi(token=token), receipt=args.receipt
            )
        else:
            data = publish_release(
                args.directory,
                api=HfApi(token=token),
                receipt=args.receipt,
                collection_slug=args.collection,
                batch_size=args.batch_size,
            )
    if args.json:
        console.json(data)
    else:
        console.kv(data, title="Harbor dataset release")
    return 0


def add_release_parser(subparsers):
    # [FUNCTION DOC] Registers the 'release' parser with actions stage (plan path, --out required),
    # verify (directory) and publish (directory, --receipt required, --collection, and mutually
    # exclusive --recover-empty / --batch-size int), each with --json and func=cmd_release. Input:
    # argparse subparsers action. Returns None; mutates the parser.
    parser = subparsers.add_parser(
        "release", help="Stage, verify and publish immutable Harbor datasets"
    )
    actions = parser.add_subparsers(dest="release_action", required=True)
    stage = actions.add_parser(
        "stage", help="Check an explicit task selection and build a release archive"
    )
    stage.add_argument("plan", type=Path)
    stage.add_argument("--out", type=Path, required=True)
    verify = actions.add_parser(
        "verify", help="Check staged bytes and modes without executing tasks"
    )
    verify.add_argument("directory", type=Path)
    publish = actions.add_parser(
        "publish", help="Upload a verified release to the Hub and optional collection"
    )
    publish.add_argument("directory", type=Path)
    publish.add_argument("--receipt", type=Path, required=True)
    publish.add_argument("--collection")
    method = publish.add_mutually_exclusive_group()
    method.add_argument(
        "--recover-empty",
        action="store_true",
        help="Confirm an uncommitted upload left an empty repository, then use bounded commits",
    )
    method.add_argument(
        "--batch-size",
        type=int,
        metavar="1..500",
        help="Publish a new empty repository in bounded commits instead of one large commit",
    )
    for command in (stage, verify, publish):
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=cmd_release)
