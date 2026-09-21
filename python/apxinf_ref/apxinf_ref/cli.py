"""``python -m apxinf_ref``: probe, infer, and compare."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import device as device_module
from . import engines
from . import infer as infer_module
from . import paths
from . import probe as probe_module

__all__ = ["main"]

ENGINES = engines.ENGINES


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "checkpoint directory (config.json + model.safetensors) or a safetensors "
            f"file. Defaults to ${paths.ENV_CHECKPOINT}, then "
            f"{paths.DEFAULT_CHECKPOINT}"
        ),
    )
    parser.add_argument("--device", default="cpu", help="cpu | cuda | musa (optionally with :index)")
    parser.add_argument(
        "--num-views",
        type=int,
        default=2,
        help="camera count; 2 matches the engine's thor_two_view profile that the stage probe uses",
    )
    parser.add_argument("--token-count", type=int, default=10)


def _inputs(config, device, *, seed: int, token_count: int, fixture: str = "zeros"):
    """The deterministic fixture.

    ``zeros`` is the default and the only one usable against the engine:
    ``crates/apxinf-model/examples/pi05_bench.rs:573-574`` rejects any
    reference whose ``normalized_images``,
    ``token_ids`` and ``diffusion_noise`` are not all ``zeros``, and the engine's
    own integrity probe feeds zeros. ``random`` exists for self-validation -- a
    zero fixture drives every stage to zero and would hide a great deal.
    """
    torch = device_module.torch_module()
    if fixture == "zeros":
        patches = torch.zeros(
            config.patch_tokens, config.patch_width, dtype=torch.float32, device=device
        )
        token_ids = torch.zeros(token_count, dtype=torch.long, device=device)
        noise = torch.zeros(
            config.action_horizon, config.action_dim, dtype=torch.float32, device=device
        )
        return patches, token_ids, noise

    generator = torch.Generator(device="cpu").manual_seed(seed)
    patches = torch.randn(
        config.patch_tokens, config.patch_width, generator=generator, dtype=torch.float32
    ).to(device)
    token_ids = torch.randint(
        0, 256, (token_count,), generator=generator, dtype=torch.long
    ).to(device)
    noise = torch.randn(
        config.action_horizon, config.action_dim, generator=generator, dtype=torch.float32
    ).to(device)
    return patches, token_ids, noise


def _resolve_precision(args, config):
    return engines.resolve_precision(args.precision, config)


def _build_engine(args, config, device, precision):
    return engines.build_engine(
        engine=args.engine,
        checkpoint=args.checkpoint,
        config=config,
        device=device,
        precision=precision,
    )


def _add_engine_choices(parser: argparse.ArgumentParser) -> None:
    """``--engine`` and ``--precision``, identical for every command that runs a model."""
    parser.add_argument("--engine", choices=ENGINES, default="assembled")
    parser.add_argument(
        "--precision",
        choices=engines.PRECISIONS,
        default="checkpoint",
        help="checkpoint follows the checkpoint's own precision (the reference "
        "definition); float32 widens weights and activations together, to tell "
        "bfloat16 arithmetic apart from the algorithm",
    )


def command_probe(args: argparse.Namespace) -> int:
    from .model.pi05.config import load_config

    device = device_module.resolve(args.device)
    device_module.manual_seed_all(device, args.seed)
    args.checkpoint = str(paths.resolve_checkpoint(args.checkpoint))
    config = load_config(args.checkpoint, num_views=args.num_views)
    precision = _resolve_precision(args, config)

    print(
        f"[apxinf_ref] engine={args.engine} device={device} "
        f"checkpoint_precision={config.precision} computing_in={precision}",
        file=sys.stderr,
    )
    model = _build_engine(args, config, device, precision)
    patches, token_ids, noise = _inputs(
        config, device, seed=args.seed, token_count=args.token_count, fixture=args.fixture
    )

    result = probe_module.run(model, patches=patches, token_ids=token_ids, noise=noise)
    print(json.dumps(result.document, indent=2, sort_keys=True))
    return 0


def command_devices(args: argparse.Namespace) -> int:
    print(device_module.describe_devices())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apxinf-ref", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    probe_parser = sub.add_parser("probe", help="emit an apxinf.pi05.stage-probe.v1 document")
    _add_common(probe_parser)
    _add_engine_choices(probe_parser)
    probe_parser.add_argument("--seed", type=int, default=0)
    probe_parser.add_argument(
        "--fixture",
        choices=("zeros", "random"),
        default="zeros",
        help="zeros matches the engine's probe; random is for self-validation",
    )
    probe_parser.set_defaults(handler=command_probe)

    infer_parser = sub.add_parser(
        "infer",
        help="run the reference on a real observation and print the stage probe for it",
    )
    _add_common(infer_parser)
    _add_engine_choices(infer_parser)
    infer_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="the noise draw's seed. Defaults to the bundle's own on a replay, and "
        "to 0 when reading LIBERO",
    )
    source = infer_parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--libero-root",
        help="directory of LIBERO demonstration HDF5 files; a frame is sampled by --seed",
    )
    source.add_argument(
        "--bundle",
        help="an apxinf.pi05.bundle.v1 directory (or its inputs.npz) to replay",
    )
    infer_parser.add_argument(
        "--tokenizer",
        default=None,
        help=f"SentencePiece model. Defaults to ${paths.ENV_TOKENIZER}, then the cache",
    )
    infer_parser.add_argument(
        "--norm-stats",
        default=None,
        help=f"norm_stats.json. Defaults to ${paths.ENV_NORM_STATS}, then the checkpoint's own",
    )
    infer_parser.add_argument(
        "--capture",
        help="also write the observation out as a bundle directory, so another host "
        "can replay it",
    )
    infer_parser.add_argument(
        "--force",
        action="store_true",
        help="with --capture, replace an existing capture instead of refusing to",
    )
    infer_parser.set_defaults(handler=infer_module.command_infer)

    compare_parser = sub.add_parser("compare", help="diff two stage-probe documents")
    compare_parser.add_argument("reference")
    compare_parser.add_argument("candidate")
    compare_parser.add_argument("--thresholds", default="bf16", choices=("bf16", "fp8", "int8"))
    compare_parser.add_argument("--json", help="also write the comparison as JSON here")
    compare_parser.add_argument("--strict", action="store_true", help="exit non-zero on failure")
    compare_parser.set_defaults(handler=_command_compare)

    devices_parser = sub.add_parser("devices", help="report what this host can run on")
    devices_parser.set_defaults(handler=command_devices)

    return parser


def _command_compare(args: argparse.Namespace) -> int:
    from .compare import compare_documents, render_table

    reference = json.loads(Path(args.reference).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    comparison = compare_documents(reference, candidate, thresholds=args.thresholds)
    print(render_table(comparison, Path(args.reference).name, Path(args.candidate).name))
    if args.json:
        Path(args.json).write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    if args.strict and not comparison["passed"]:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
