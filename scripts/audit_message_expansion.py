"""Independently rescore train OOF/dev, verify frozen sources, and confirm deployment hold."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml
from sklearn.model_selection import StratifiedGroupKFold

from cti.core.sms_data import ROOT, SPLITS
from cti.core.sms_domain import SPLIT, load_domain
from cti.experiment_message import activate, choose, digest, original_rows
from cti.models.message_v2 import load_message_classifier


def independent_metrics(rows, predictions):
    result = {}
    for source in ("uci", "mendeley"):
        pairs = [(r["label"] == "spam", p == "spam") for r, p in zip(rows, predictions) if r["source"] == source]
        tp = sum(g and p for g, p in pairs); fp = sum(not g and p for g, p in pairs)
        fn = sum(g and not p for g, p in pairs); tn = sum(not g and not p for g, p in pairs)
        result[source] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                          "precision": tp / (tp + fp) if tp + fp else 0.0,
                          "recall": tp / (tp + fn) if tp + fn else 0.0,
                          "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    return {"per_source": result, "source_mean_f1": sum(r["f1"] for r in result.values()) / 2}


def same(actual, expected):
    if isinstance(actual, dict):
        assert set(actual) == set(expected)
        for key in actual:
            same(actual[key], expected[key])
    else:
        assert math.isclose(actual, expected, abs_tol=1e-12), (actual, expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/message_generalization_v1")
    args = parser.parse_args(argv)
    folder = args.results_dir
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    config_path = ROOT / "configs/message_expansion.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert digest(config_path) == selection["provenance"]["config_sha256"]
    assert digest(SPLITS) == selection["provenance"]["uci_split_sha256"]
    assert digest(SPLIT) == selection["provenance"]["domain_split_sha256"]
    for name, expected in selection["provenance"]["source_sha256"].items():
        assert digest(ROOT / name) == expected, name
    rows = original_rows("train") + load_domain("train")
    folds = list(StratifiedGroupKFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"]).split(
        rows, [r["source"] + ":" + r["original_label"] for r in rows], [r["group"] for r in rows]))
    for candidate in selection["candidates"]:
        cached = json.loads((folder / (candidate["id"] + "_oof.json")).read_text(encoding="utf-8"))
        assert [(r["id"], r["label"]) for r in rows] == [(r["id"], r["gold"]) for r in cached]
        predictions = [r["predicted"] for r in cached]
        same(independent_metrics(rows, predictions), candidate["pooled"])
        for (fit_ids, valid_ids), stored in zip(folds, candidate["folds"]):
            assert {rows[i]["group"] for i in fit_ids}.isdisjoint(rows[i]["group"] for i in valid_ids)
            same(independent_metrics([rows[i] for i in valid_ids], [predictions[i] for i in valid_ids]), stored)
    assert choose(selection["candidates"], config["selection"]) == summary["selected"]
    dev = original_rows("dev") + load_domain("dev")
    active_path = ROOT / "models/message_spam.joblib"
    assert hashlib.sha256(active_path.read_bytes()).hexdigest() == summary["previous_active_sha256"]
    active = load_message_classifier(active_path)
    candidate = load_message_classifier(ROOT / summary["candidate_path"])
    assert candidate.manifest["artifact_sha256"] == summary["candidate_sha256"]
    cached = json.loads((folder / "dev_predictions.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in dev] == [r["id"] for r in cached]
    for name, artifact in (("baseline", active), ("candidate", candidate)):
        predicted = artifact.pipeline.predict([r["text"] for r in dev]).tolist()
        assert predicted == [r[name] for r in cached]
        same(independent_metrics(dev, predicted), summary[name + "_dev"])
    challenge = json.loads((folder / "challenge.json").read_text(encoding="utf-8"))
    authored = json.loads((ROOT / "examples/message_challenge.json").read_text(encoding="utf-8"))
    assert authored["messages"] == [{k: v for k, v in r.items() if k not in ("baseline", "candidate")} for r in challenge["cases"]]
    for name, artifact in (("baseline", active), ("candidate", candidate)):
        predicted = artifact.pipeline.predict([r["text"] for r in authored["messages"]]).tolist()
        assert predicted == [r[name] for r in challenge["cases"]]
        # Reuse only the independent counting formulas, treating this diagnostic
        # as one source; no challenge examples enter training or CV selection.
        diagnostic = [{**r, "source": "mendeley"} for r in authored["messages"]]
        same(independent_metrics(diagnostic, predicted)["per_source"]["mendeley"], challenge[name])
    reported = "PM Modi is giving 15 lakhs rupees to every citizen of our country . To credit it in you bank account .\nKindly SMS 6969 on following number :\n6969696969"
    probes = []
    for variant, text in (("original", reported), ("shortened_target", reported[:-1]),
                          ("missing_target", reported.rsplit("\n", 1)[0])):
        decision = candidate.predict_one(text)
        probes.append({"variant": variant, "label": decision["label"], "decision_margin": decision["decision_margin"]})
    assert summary["deployment_hold"] and challenge["candidate"]["fp"] > challenge["baseline"]["fp"]
    try:
        activate(folder)
    except ValueError as error:
        assert "held" in str(error)
    else:
        raise AssertionError("A held candidate must not replace the default.")
    assert hashlib.sha256(active_path.read_bytes()).hexdigest() == summary["previous_active_sha256"]
    result = {"all_12_fold_metrics_reproduced": True, "source_hashes_verified": True,
              "dev_predictions_reloaded_and_rescored": True, "selection_train_only": True,
              "default_artifact_unchanged": True, "deployment_hold_enforced": True,
              "challenge_not_used_for_fit_or_cv_selection": True, "test_evaluated": False,
              "reported_message_candidate_probes": probes,
              "audit_source_sha256": digest(__file__)}
    (folder / "integrity_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
