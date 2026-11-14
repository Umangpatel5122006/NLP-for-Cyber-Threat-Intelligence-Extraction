"""Analyze a text report with a configured classical NLP pipeline."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from cti.core.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config_ner.yaml")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="Report text to analyze.")
    inputs.add_argument("--input", type=Path, help="UTF-8 plain-text report file.")
    parser.add_argument("--output", type=Path, help="Write JSON to this file instead of stdout.")
    args = parser.parse_args(argv)
    text = args.input.read_text(encoding="utf-8") if args.input is not None else args.text
    document = Pipeline.from_config(args.config).run(text)
    result = json.dumps(asdict(document), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
        print(f"Saved analysis: {args.output}")


if __name__ == "__main__":
    main()
