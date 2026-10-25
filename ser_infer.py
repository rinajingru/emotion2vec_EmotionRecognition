#!/usr/bin/env python3
"""Command-line speech emotion recognition with emotion2vec+.

The script intentionally keeps the emotion2vec encoder frozen and delegates
audio loading/model management to the official FunASR interface.  It accepts
one 16 kHz speech recording and prints a JSON object containing the predicted
emotion and all returned scores.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Any


LABELS = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "other",
    6: "sad",
    7: "surprised",
    8: "unknown",
}


def _first_record(result: Any) -> dict[str, Any]:
    """Normalize FunASR's dict-or-list return value."""
    if isinstance(result, list):
        if not result:
            raise RuntimeError("FunASR returned an empty result.")
        result = result[0]
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected FunASR result type: {type(result)!r}")
    return result


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    # NumPy and Torch tensors are deliberately handled without importing them.
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    return [value]


def _label_name(label: Any) -> str:
    if isinstance(label, int):
        return LABELS.get(label, str(label))
    if isinstance(label, float) and label.is_integer():
        return LABELS.get(int(label), str(int(label)))
    text = str(label).strip()
    # Some versions return labels as "0: angry".
    if ":" in text and text.split(":", 1)[0].strip().isdigit():
        text = text.split(":", 1)[1].strip()
    # FunASR's multilingual label table can return strings such as
    # ``生气/angry``; retain the stable English suffix for the JSON schema.
    if "/" in text:
        suffix = text.rsplit("/", 1)[1].strip().lower()
        if suffix in LABELS.values():
            return suffix
    if text.lower() in {"<unk>", "unk"}:
        return "unknown"
    return LABELS.get(int(text), text) if text.isdigit() else text


def _parse_scores(record: dict[str, Any]) -> dict[str, float]:
    labels = [_label_name(x) for x in _as_list(record.get("labels"))]
    scores = _as_list(record.get("scores"))
    # A few FunASR releases return a single nested row for an utterance.
    if len(scores) == 1 and isinstance(scores[0], (list, tuple)):
        scores = list(scores[0])
    if not labels and scores:
        labels = [LABELS.get(i, str(i)) for i in range(len(scores))]
    if len(labels) != len(scores):
        raise RuntimeError(
            "Model returned mismatched labels/scores lengths: "
            f"{len(labels)} vs {len(scores)}"
        )
    parsed: dict[str, float] = {}
    for label, score in zip(labels, scores):
        if hasattr(score, "item"):
            score = score.item()
        parsed[label] = float(score)
    return parsed


def _audio_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except (wave.Error, OSError, ZeroDivisionError):
        # FunASR can decode formats supported by soundfile/ffmpeg; the stdlib
        # wave reader is only used for a friendly metadata field.
        return None


def predict(
    audio_path: str | Path,
    *,
    model_id: str = "iic/emotion2vec_plus_seed",
    hub: str = "ms",
    output_dir: str | Path = "outputs",
    extract_embedding: bool = False,
) -> dict[str, Any]:
    """Run one utterance through the official emotion2vec+ checkpoint."""
    audio = Path(audio_path).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio}")
    try:
        from funasr import AutoModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "FunASR is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc

    model = AutoModel(model=model_id, hub=hub, disable_update=True)
    return _predict_with_model(
        model,
        audio,
        model_id=model_id,
        hub=hub,
        output_dir=output_dir,
        extract_embedding=extract_embedding,
    )


def _predict_with_model(
    model: Any,
    audio: Path,
    *,
    model_id: str,
    hub: str,
    output_dir: str | Path,
    extract_embedding: bool,
) -> dict[str, Any]:
    """Run inference using an already-loaded AutoModel instance."""
    raw = model.generate(
        str(audio),
        output_dir=str(output_dir),
        granularity="utterance",
        extract_embedding=extract_embedding,
    )
    record = _first_record(raw)
    scores = _parse_scores(record)
    if scores:
        top_label, top_score = max(scores.items(), key=lambda item: item[1])
    else:
        top_label, top_score = "unknown", None
    output: dict[str, Any] = {
        "audio": str(audio),
        "model": model_id,
        "hub": hub,
        "sample_rate_hz": 16000,
        "top_emotion": top_label,
        "confidence": top_score,
        "scores": dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)),
    }
    duration = _audio_duration(audio)
    if duration is not None:
        output["duration_sec"] = round(duration, 3)
    if extract_embedding and "feats" in record:
        feats = record["feats"]
        shape = getattr(feats, "shape", None)
        output["embedding_shape"] = list(shape) if shape is not None else None
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict the emotion expressed in one speech recording."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        help="Path to a WAV/other audio file. Omit it with --self-test.",
    )
    parser.add_argument(
        "--model",
        default="iic/emotion2vec_plus_seed",
        help="FunASR emotion2vec checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--hub",
        default="ms",
        choices=("hf", "huggingface", "ms", "modelscope"),
        help="Model hub used for automatic download (default: %(default)s).",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--save-json",
        help="Optional path at which to save the JSON prediction.",
    )
    parser.add_argument(
        "--extract-embedding",
        action="store_true",
        help="Also request the utterance embedding from FunASR.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Use the checkpoint's bundled example/test.wav after download.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.self_test and args.audio:
        raise SystemExit("Pass either AUDIO or --self-test, not both.")
    if args.self_test:
        # Loading the model first gives us the official example audio path.
        try:
            from funasr import AutoModel  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "FunASR is not installed. Run `python -m pip install -r requirements.txt`."
            ) from exc
        model = AutoModel(model=args.model, hub=args.hub, disable_update=True)
        example = Path(model.model_path) / "example" / "test.wav"
        if not example.is_file():
            raise SystemExit(f"The downloaded checkpoint has no example audio: {example}")
        audio = example
    elif args.audio:
        audio = Path(args.audio)
    else:
        raise SystemExit("Provide AUDIO or use --self-test.")

    if args.self_test:
        result = _predict_with_model(
            model,
            audio,
            model_id=args.model,
            hub=args.hub,
            output_dir=args.output_dir,
            extract_embedding=args.extract_embedding,
        )
    else:
        result = predict(
            audio,
            model_id=args.model,
            hub=args.hub,
            output_dir=args.output_dir,
            extract_embedding=args.extract_embedding,
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.save_json:
        destination = Path(args.save_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
