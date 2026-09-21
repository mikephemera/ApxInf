"""``python -m apxinf_ref infer``: the reference on a real observation.

The stage probe answers "what is the right number at each layer" for a fixture
both sides can produce. This command answers the question in front of that one:
*given an observation*, what does the reference do? It builds the model's three
inputs from a LIBERO demonstration or a captured bundle, runs the same forward
pass the probe runs, and writes both the stage probe for that observation and the
action chunk it produced.

The document keeps the stage-probe schema, so ``compare`` reads it unchanged and
the same per-stage subtraction works against an engine run on the same
observation. What it adds is the actions and the provenance that makes them
attributable -- which frame, whether it was rotated, which resize, which
normalization statistics, and which noise. Those are the facts that decide
whether two runs are comparable, and none of them is inferable from the numbers.

``--capture`` writes the observation out as a bundle, which is the other half of
the exchange. A probe takes patches and token ids directly, so a probe taken on
one host says nothing about a probe taken on another unless both were handed the
same inputs; a bundle is that input, written where the observation is and
replayed wherever the port needs a comparison. ``--capture`` and ``--out`` are
independent: a capture that recorded a probe but no bundle, or a bundle with no
standalone probe, would each be a way to lose half the result.

The noise is checked element by element on a replay, and a mismatch is refused.
That is deliberately the one hard stop: noise is an *input*, so a replay under a
different draw answers a different question while looking exactly like the same
one. Everything else a bundle records is reported and changes no exit code -- a
differing artifact digest, a differing set of token ids, and the thresholds the
producer was willing to call agreement all go into the document. Two hosts whose
tokenizer files differ produce different token ids and their stage signatures are
not comparable, but that is a fact for the reader rather than grounds for a
refusal.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import bundle as bundle_module
from . import compare as compare_module
from . import device as device_module
from . import engines
from . import paths
from . import probe as probe_module
from . import sources
from .model.pi05.tokenizer import sha256_of

__all__ = ["command_infer"]

#: The schema ``compare`` reads. An infer document *is* a stage probe, with the
#: actions and the provenance attached, so one comparison tool serves both.
SCHEMA = probe_module.SCHEMA

#: What this project calls agreement, recorded in a bundle and never enforced by
#: it. The reference runs in the checkpoint's own precision and a bundle exists to
#: be subtracted from a bfloat16 port, so the bfloat16 set is the one worth
#: writing down; the run's own precision is recorded separately, in
#: ``producer.precision``.
THRESHOLDS = {
    "stage_cosine": compare_module.THRESHOLDS["bf16"]["min_cosine"],
    "final_actions_cosine": compare_module.THRESHOLDS["bf16"]["min_cosine"],
}


def _source(args: argparse.Namespace) -> sources.Source:
    if args.libero_root:
        # A LIBERO read samples its frame by seed, so the seed has to be settled
        # before the source exists; 0 is the documented capture seed.
        return sources.from_libero(
            args.libero_root, seed=0 if args.seed is None else int(args.seed)
        )
    return sources.from_bundle(args.bundle)


def _resolve_seed(args: argparse.Namespace, source: sources.Source) -> int:
    """The seed this run derives its noise from.

    A replay borrows the one its bundle recorded unless it was told otherwise.
    The documented replay command passes no ``--seed``, and a capture made under
    a different one would otherwise be refused by the noise check for a reason
    that has nothing to do with what is being compared.
    """
    if args.seed is not None:
        return int(args.seed)
    recorded = (source.extras.get("manifest") or {}).get("seed")
    return int(recorded) if recorded is not None else 0


def _refuse_a_bundle_that_asks_another_question(
    source: sources.Source, *, config, inputs
) -> None:
    """A bundle captured at another view count, image size or prompt length.

    These are not facts about a host, they are facts about the question, and a
    disagreement means every stage downstream would be compared under a name
    that claims the two runs asked the same thing. Refusing here says which
    number differs; letting it through would surface much later as a shape error
    or, worse, as a cosine that looks like a precision difference.
    """
    manifest = source.extras.get("manifest")
    if not manifest:
        return
    recorded = manifest.get("source") or {}
    checks = (
        ("view count", recorded.get("num_views"), int(config.num_views)),
        ("image size", recorded.get("image_size"), int(config.image_size)),
        ("token count", manifest.get("token_count"), int(inputs.token_count)),
    )
    for what, theirs, ours in checks:
        if theirs is not None and int(theirs) != ours:
            raise SystemExit(
                f"refused: the bundle records {what} {int(theirs)} and this run has "
                f"{ours}. The prefix length and every stage downstream follow it, so "
                "the two documents would be compared as if they had asked the same "
                "question. Re-run with the matching configuration, or capture a "
                "bundle for this one."
            )


def _reconcile_noise(inputs, frozen, device):
    """Insist the frozen noise the bundle carries is the one this run derives.

    The noise is an *input*, not a rounding difference: a bundle replayed under a
    different draw answers a different question while looking like the same one.
    """
    if frozen is None:
        return inputs
    torch = device_module.torch_module()
    ours = inputs.noise.detach().to("cpu").float().numpy()
    if tuple(frozen.shape) != tuple(ours.shape) or not np.array_equal(frozen, ours):
        raise SystemExit(
            f"refused: the bundle's frozen noise is not the one this seed derives "
            f"(bundle {tuple(frozen.shape)}, derived {tuple(ours.shape)}; "
            f"max abs difference "
            f"{float(np.abs(np.asarray(frozen) - ours).max()) if frozen.shape == ours.shape else float('nan')}). "
            "Replay it with the --seed it was captured under."
        )
    return dataclasses.replace(inputs, noise=torch.from_numpy(np.asarray(frozen)).to(device))


def _artifact_digests(*, checkpoint, tokenizer_path, tokenizer_digest, norm_stats_path):
    """The three files that decide whether two runs are comparable at all.

    The checkpoint's digest is the expensive one -- a PI0.5 export is several
    gigabytes, so this reads the whole file. It is deliberately not cached: a
    digest keyed on size and modification time is a digest that can silently go
    stale, and one that is usually right is worth less than none at all.
    """
    weights = Path(checkpoint) / paths.WEIGHTS_NAME
    started = time.perf_counter()
    digests = {
        "checkpoint": {
            "path": str(checkpoint),
            "weights": paths.WEIGHTS_NAME,
            "sha256": sha256_of(weights),
        },
        "tokenizer": {"path": str(tokenizer_path), "sha256": tokenizer_digest},
        "norm_stats": {"path": str(norm_stats_path), "sha256": sha256_of(norm_stats_path)},
    }
    elapsed = time.perf_counter() - started
    if elapsed >= 1.0:
        print(
            f"[apxinf_ref] artifact digests in {elapsed:.1f} s "
            f"({weights.stat().st_size / 1e9:.2f} GB checkpoint)",
            file=sys.stderr,
        )
    return digests


def _receipt(*, args, device, precision, source, seed, inputs, config, checkpoint, artifacts):
    """The run's facts, as lines for a person -- stderr now, ``run.log`` in a bundle."""
    return [
        f"engine={args.engine} device={device} precision={precision}",
        f"source={source.kind} seed={seed} views={int(config.num_views)} "
        f"tokens={inputs.token_count} state={inputs.state_width} "
        f"image_size={inputs.image_size}",
        f"source_image_size={tuple(inputs.source_image_size)} resize={source.resize}",
        f"prompt={inputs.prompt!r}",
        f"checkpoint={checkpoint}",
    ] + [
        f"artifacts.{name}={entry['sha256']} ({entry['path']})"
        for name, entry in artifacts.items()
    ]


def _document(
    *,
    inputs,
    result,
    args,
    device,
    config,
    checkpoint,
    precision,
    seed,
    source,
    raw,
    actions,
):
    document = dict(result.document)
    document.update(
        {
            "actions": [[float(value) for value in row] for row in actions],
            "raw_actions": [[float(value) for value in row] for row in raw],
            "checkpoint": str(checkpoint),
            "device": str(device),
            "engine": args.engine,
            "image_size": inputs.image_size,
            "num_views": int(config.num_views),
            # A recorded fact, not an inference from which kind of source this
            # was: once a bundle crosses a machine boundary the question is about
            # another host's behaviour, and the producer is the one who answered
            # it. See `sources.py`.
            "oriented": bool(source.oriented),
            "precision": precision,
            "prompt": inputs.prompt,
            "resize": source.resize,
            "seed": int(seed),
            "source": dict(source.provenance),
            "state_width": inputs.state_width,
            "token_count": inputs.token_count,
        }
    )
    return document


def command_infer(args: argparse.Namespace) -> int:
    from .model.pi05 import observation as observation_module
    from .model.pi05.config import load_config

    if args.capture and args.bundle:
        raise SystemExit(
            "--capture cannot be combined with --bundle: a capture records what this "
            "host observed, and a replay observes nothing. Its answer would be the "
            "replay's own actions wearing the name of a reference."
        )
    if args.force and not args.capture:
        raise SystemExit("--force only means something with --capture: it replaces a capture.")

    checkpoint = paths.resolve_checkpoint(args.checkpoint)
    config = load_config(checkpoint, num_views=args.num_views)
    if config.num_views != 2:
        raise SystemExit(
            f"infer needs --num-views 2: a LIBERO observation and a captured bundle "
            f"each carry two cameras (base and wrist), and {config.num_views} was "
            "requested."
        )

    source = _source(args)
    seed = _resolve_seed(args, source)
    device = device_module.resolve(args.device)
    device_module.manual_seed_all(device, seed)
    precision = engines.resolve_precision(args.precision, config)

    tokenizer_path = paths.resolve_tokenizer(checkpoint, args.tokenizer)
    norm_stats_path = paths.resolve_norm_stats(checkpoint, args.norm_stats)
    norm_stats = observation_module.read_norm_stats(norm_stats_path)

    inputs = observation_module.assemble(
        source.observation,
        config=config,
        tokenizer_path=tokenizer_path,
        norm_stats=norm_stats,
        device=device,
        seed=seed,
    )
    _refuse_a_bundle_that_asks_another_question(source, config=config, inputs=inputs)
    inputs = _reconcile_noise(inputs, source.extras.get("frozen_noise"), device)

    artifacts = _artifact_digests(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_path,
        tokenizer_digest=inputs.tokenizer_digest,
        norm_stats_path=norm_stats_path,
    )
    receipt = _receipt(
        args=args,
        device=device,
        precision=precision,
        source=source,
        seed=seed,
        inputs=inputs,
        config=config,
        checkpoint=checkpoint,
        artifacts=artifacts,
    )
    for line in receipt:
        print(f"[apxinf_ref] {line}", file=sys.stderr)

    model = engines.build_engine(
        engine=args.engine,
        checkpoint=checkpoint,
        config=config,
        device=device,
        precision=precision,
    )
    result = probe_module.run(
        model, patches=inputs.patches, token_ids=inputs.token_ids, noise=inputs.noise
    )

    raw = result.action.detach().to("cpu").float().numpy()
    actions = observation_module.final_actions(raw, norm_stats=norm_stats)
    document = _document(
        inputs=inputs,
        result=result,
        args=args,
        device=device,
        config=config,
        checkpoint=checkpoint,
        precision=precision,
        seed=seed,
        source=source,
        raw=raw,
        actions=actions,
    )

    manifest = source.extras.get("manifest")
    if manifest:
        document["bundle"] = bundle_module.replay_report(
            manifest,
            local_artifacts=artifacts,
            token_ids=source.extras.get("token_ids"),
        )
        _report_a_bundle_that_disagrees(document["bundle"])

    gold = source.extras.get("gold_actions")
    if gold is not None:
        document["gold_cosine"] = _cosine(actions, gold)

    if args.capture:
        destination = bundle_module.write_bundle(
            args.capture,
            observation=source.observation,
            inputs=inputs,
            document=document,
            actions=actions,
            raw_actions=raw,
            source=source,
            artifacts=artifacts,
            thresholds=THRESHOLDS,
            engine=args.engine,
            device=device,
            precision=precision,
            seed=seed,
            receipt=receipt,
            force=args.force,
        )
        print(f"[apxinf_ref] captured {destination}", file=sys.stderr)

    if args.out:
        written = probe_module.write(document, args.out)
        print(f"[apxinf_ref] wrote {written}", file=sys.stderr)
    else:
        print(json.dumps(document, indent=2, sort_keys=True))
    return 0


def _report_a_bundle_that_disagrees(report) -> None:
    """Say out loud what the document records, and change no exit code.

    A mismatch here means the two hosts are not comparing the same thing, which
    is worth interrupting a terminal for and is not worth refusing to run: the
    reader of the document decides what it means.
    """
    for name, entry in sorted(report.get("artifacts_match", {}).items()):
        if not entry["matched"]:
            print(
                f"[apxinf_ref] the bundle's {name} digest is {entry['recorded']} and this "
                f"host's is {entry['local']}; the two runs did not use the same file",
                file=sys.stderr,
            )
    if report.get("token_ids_match") is False:
        counts = report["token_count"]
        print(
            f"[apxinf_ref] the bundle's prompt discretised into {counts['recorded']} token "
            f"ids and this host's into {counts['local']}; the two documents are not "
            "comparable stage by stage",
            file=sys.stderr,
        )


def _cosine(actions, gold) -> float:
    """Cosine of the produced chunk against a bundle's recorded gold actions.

    Reported, never gated: this command is the reference, so it has nothing to
    pass against. A number here is a statement about agreement, and a reader
    decides what it means.
    """
    a = np.asarray(actions, dtype=np.float64).reshape(-1)
    b = np.asarray(gold, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return float("nan")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else float("nan")
