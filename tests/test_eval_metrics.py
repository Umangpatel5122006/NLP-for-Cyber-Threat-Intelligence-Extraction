import csv
import itertools
import tempfile
import unittest
from pathlib import Path

from cti.eval.metrics import (
    classify_errors,
    evaluate_predictions,
    exact_span_metrics,
    extract_spans,
    relaxed_span_metrics,
    token_level_metrics,
)


class EvalMetricsTests(unittest.TestCase):
    def test_extract_spans_extracts_correct_half_open_intervals(self):
        seq = ["O", "B-ACTOR", "I-ACTOR", "O", "B-TOOL"]
        spans = extract_spans(seq)
        self.assertEqual(spans, [("ACTOR", 1, 3), ("TOOL", 4, 5)])

    def test_exact_match_scores_perfect_f1(self):
        gold = [["B-THREAT_ACTOR", "I-THREAT_ACTOR", "O", "B-MALWARE_TOOL"]]
        pred = [["B-THREAT_ACTOR", "I-THREAT_ACTOR", "O", "B-MALWARE_TOOL"]]

        metrics = exact_span_metrics(gold, pred)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["per_label"]["THREAT_ACTOR"]["f1"], 1.0)
        self.assertEqual(metrics["per_label"]["MALWARE_TOOL"]["f1"], 1.0)

        err = classify_errors(gold, pred)
        self.assertEqual(err["counts"]["exact_match"], 2)
        self.assertEqual(err["counts"]["boundary_error"], 0)
        self.assertEqual(err["counts"]["wrong_type"], 0)
        self.assertEqual(err["counts"]["missed"], 0)
        self.assertEqual(err["counts"]["spurious"], 0)

    def test_boundary_error_fails_exact_but_passes_relaxed(self):
        # Gold entity spans tokens 0..2 ("Cobalt Strike")
        # Predicted entity spans only token 1 ("Strike")
        gold = [["B-MALWARE_TOOL", "I-MALWARE_TOOL", "O"]]
        pred = [["O", "B-MALWARE_TOOL", "O"]]

        exact = exact_span_metrics(gold, pred)
        self.assertEqual(exact["f1"], 0.0)

        relaxed = relaxed_span_metrics(gold, pred)
        self.assertEqual(relaxed["precision"], 1.0)
        self.assertEqual(relaxed["recall"], 1.0)
        self.assertEqual(relaxed["f1"], 1.0)

        err = classify_errors(gold, pred)
        self.assertEqual(err["counts"]["exact_match"], 0)
        self.assertEqual(err["counts"]["boundary_error"], 1)
        self.assertEqual(err["counts"]["wrong_type"], 0)
        self.assertEqual(err["counts"]["missed"], 0)
        self.assertEqual(err["counts"]["spurious"], 0)

    def test_type_confusion_identified_correctly(self):
        # Gold is MALWARE_TOOL, prediction overlaps but is labeled INFRASTRUCTURE
        gold = [["B-MALWARE_TOOL", "O"]]
        pred = [["B-INFRASTRUCTURE", "O"]]

        exact = exact_span_metrics(gold, pred)
        self.assertEqual(exact["f1"], 0.0)

        relaxed = relaxed_span_metrics(gold, pred)
        self.assertEqual(relaxed["f1"], 0.0)

        err = classify_errors(gold, pred)
        self.assertEqual(err["counts"]["exact_match"], 0)
        self.assertEqual(err["counts"]["boundary_error"], 0)
        self.assertEqual(err["counts"]["wrong_type"], 1)
        self.assertEqual(err["counts"]["missed"], 0)
        self.assertEqual(err["counts"]["spurious"], 0)
        self.assertEqual(err["type_confusion_pairs"]["MALWARE_TOOL -> INFRASTRUCTURE"], 1)

    def test_completely_missed_and_completely_spurious(self):
        gold = [["B-THREAT_ACTOR", "O", "O"]]
        pred = [["O", "O", "B-INFRASTRUCTURE"]]

        exact = exact_span_metrics(gold, pred)
        self.assertEqual(exact["precision"], 0.0)
        self.assertEqual(exact["recall"], 0.0)
        self.assertEqual(exact["f1"], 0.0)

        relaxed = relaxed_span_metrics(gold, pred)
        self.assertEqual(relaxed["f1"], 0.0)

        err = classify_errors(gold, pred)
        self.assertEqual(err["counts"]["missed"], 1)
        self.assertEqual(err["counts"]["spurious"], 1)
        self.assertEqual(err["counts"]["exact_match"], 0)

    def test_token_level_evaluation(self):
        gold = [["B-ACTOR", "I-ACTOR", "O"]]
        pred = [["B-ACTOR", "O", "O"]]

        tokens = token_level_metrics(gold, pred)
        self.assertEqual(tokens["per_label"]["ACTOR"]["support"], 2)
        self.assertEqual(tokens["per_label"]["ACTOR"]["precision"], 1.0)
        self.assertEqual(tokens["per_label"]["ACTOR"]["recall"], 0.5)
        self.assertAlmostEqual(tokens["per_label"]["ACTOR"]["f1"], 0.6667, places=3)

    def test_missing_all_entities_gets_no_credit_for_background_tokens(self):
        metrics = token_level_metrics(
            [["B-ACTOR", "O", "O", "O"]], [["O", "O", "O", "O"]]
        )
        self.assertNotIn("O", metrics["per_label"])
        self.assertEqual(metrics["micro_f1"], 0.0)
        self.assertEqual(metrics["macro_f1"], 0.0)

    def test_entity_prediction_on_background_is_a_false_positive(self):
        metrics = token_level_metrics(
            [["B-ACTOR", "O"]], [["B-ACTOR", "B-ACTOR"]]
        )
        self.assertEqual(metrics["micro_precision"], 0.5)
        self.assertEqual(metrics["micro_recall"], 1.0)
        self.assertAlmostEqual(metrics["micro_f1"], 2 / 3, places=4)

    def test_predicted_only_entity_types_are_scored(self):
        metrics = token_level_metrics([["O"]], [["B-TOOL"]])
        self.assertEqual(set(metrics["per_label"]), {"TOOL"})
        self.assertEqual(metrics["per_label"]["TOOL"]["support"], 0)
        self.assertEqual(metrics["micro_f1"], 0.0)

    def test_empty_token_scores_have_the_same_schema(self):
        keys = set(token_level_metrics([["B-ACTOR"]], [["B-ACTOR"]]))
        for gold, predictions in (([], []), ([[]], [[]]), ([["O"]], [["O"]])):
            with self.subTest(gold=gold):
                metrics = token_level_metrics(gold, predictions)
                self.assertEqual(set(metrics), keys)
                self.assertEqual(metrics["per_label"], {})
                for key in keys - {"per_label"}:
                    self.assertEqual(metrics[key], 0.0)

    def test_one_prediction_cannot_recall_two_gold_spans(self):
        metrics = relaxed_span_metrics(
            [["B-ACTOR", "O", "B-ACTOR"]],
            [["B-ACTOR", "I-ACTOR", "I-ACTOR"]],
        )
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 2 / 3, places=4)
        self.assertEqual(metrics["per_label"]["ACTOR"]["true_positives"], 1)

    def test_two_predictions_cannot_both_match_one_gold_span(self):
        metrics = relaxed_span_metrics(
            [["B-ACTOR", "I-ACTOR", "I-ACTOR"]],
            [["B-ACTOR", "O", "B-ACTOR"]],
        )
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 2 / 3, places=4)

    def test_relaxed_matching_recovers_multiple_distinct_pairs(self):
        metrics = relaxed_span_metrics(
            [["B-ACTOR", "I-ACTOR", "I-ACTOR", "B-ACTOR"]],
            [["B-ACTOR", "O", "B-ACTOR", "I-ACTOR"]],
        )
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["per_label"]["ACTOR"]["true_positives"], 2)

    def test_relaxed_matching_does_not_cross_sentence_boundaries(self):
        metrics = relaxed_span_metrics(
            [["B-ACTOR"], ["O"]], [["O"], ["B-ACTOR"]]
        )
        self.assertEqual(metrics["f1"], 0.0)

    def test_relaxed_matching_agrees_with_exhaustive_pairing(self):
        sequences = []
        for tags in itertools.product(("O", "B-ACTOR", "I-ACTOR"), repeat=4):
            if all(
                tag != "I-ACTOR" or (index > 0 and tags[index - 1] != "O")
                for index, tag in enumerate(tags)
            ):
                sequences.append(list(tags))

        def best_pairing(gold_spans, predicted_spans, index=0, used=frozenset()):
            if index == len(gold_spans):
                return 0
            best = best_pairing(gold_spans, predicted_spans, index + 1, used)
            _, start, end = gold_spans[index]
            for p_index, (_, p_start, p_end) in enumerate(predicted_spans):
                if p_index not in used and max(start, p_start) < min(end, p_end):
                    best = max(best, 1 + best_pairing(
                        gold_spans, predicted_spans, index + 1, used | {p_index}
                    ))
            return best

        for gold in sequences:
            for predictions in sequences:
                expected = best_pairing(extract_spans(gold), extract_spans(predictions))
                metrics = relaxed_span_metrics([gold], [predictions])
                actual = metrics["per_label"].get("ACTOR", {}).get("true_positives", 0)
                self.assertEqual(actual, expected, (gold, predictions))

    def test_strict_per_label_counts_follow_strict_iob2(self):
        gold, predictions = [["B-ACTOR", "O"]], [["I-ACTOR", "O"]]
        default = exact_span_metrics(gold, predictions)
        strict = exact_span_metrics(gold, predictions, mode="strict")
        self.assertEqual(default["f1"], 1.0)
        self.assertEqual(strict["f1"], 0.0)
        self.assertEqual(strict["per_label"]["ACTOR"]["predicted"], 0)
        self.assertEqual(strict["per_label"]["ACTOR"]["true_positives"], 0)
        self.assertEqual(strict["per_label"]["ACTOR"]["f1"], 0.0)

    def test_exact_scoring_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            exact_span_metrics([["O"]], [["O"]], mode="typo")

    def test_metrics_reject_misaligned_sequences(self):
        functions = (
            exact_span_metrics, relaxed_span_metrics, token_level_metrics,
            classify_errors, evaluate_predictions,
        )
        cases = (
            ([["O"]], []),
            ([["B-ACTOR"], ["O"]], [["B-ACTOR", "O"], []]),
        )
        for function in functions:
            for gold, predictions in cases:
                with self.subTest(function=function.__name__, gold=gold):
                    with self.assertRaises(ValueError):
                        function(gold, predictions)

    def test_empty_evaluation_bundle_has_zero_scores(self):
        for gold in ([], [[]], [["O"]]):
            with self.subTest(gold=gold):
                bundle = evaluate_predictions(gold, gold)
                for key in ("exact_default", "exact_strict", "relaxed"):
                    self.assertEqual(bundle[key]["f1"], 0.0)
                self.assertEqual(bundle["token_level"]["macro_f1"], 0.0)
                self.assertEqual(bundle["error_taxonomy"]["total_gold"], 0)

    def test_evaluate_predictions_bundle(self):
        gold = [["B-ACTOR", "O"], ["O", "B-TOOL"]]
        pred = [["B-ACTOR", "O"], ["O", "O"]]

        bundle = evaluate_predictions(gold, pred)
        self.assertIn("exact_default", bundle)
        self.assertIn("exact_strict", bundle)
        self.assertIn("relaxed", bundle)
        self.assertIn("token_level", bundle)
        self.assertIn("error_taxonomy", bundle)
        self.assertEqual(bundle["error_taxonomy"]["counts"]["exact_match"], 1)
        self.assertEqual(bundle["error_taxonomy"]["counts"]["missed"], 1)


class EvaluationReportTests(unittest.TestCase):
    def make_summary(self, **overrides):
        from cti.eval.report import generate_evaluation_summary

        arguments = {
            "model_name": "synthetic",
            "layer": "entities",
            "gold": [["B-ACTOR", "O"]],
            "predictions": [["O", "O"]],
            "doc_ids": ["synthetic-report"],
            "sources": ["synthetic-source"],
            "relevant_mask": [True],
        }
        arguments.update(overrides)
        return generate_evaluation_summary(**arguments)

    def test_report_exports_corrected_scores_and_metrics_version(self):
        from cti.eval.report import export_report_csv, export_report_markdown

        summary = self.make_summary()
        self.assertEqual(summary["metrics_version"], "2")
        self.assertEqual(summary["per_source"]["synthetic-source"]["token_macro_f1"], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "report.csv"
            md_path = Path(directory) / "report.md"
            export_report_csv(summary, csv_path)
            export_report_markdown(summary, md_path)
            with csv_path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(rows)
            self.assertTrue(all(row["metrics_version"] == "2" for row in rows))
            token = next(row for row in rows if row["metric_mode"] == "token_macro")
            self.assertEqual(float(token["f1"]), 0.0)
            self.assertIn("Metrics version", md_path.read_text(encoding="utf-8"))

    def test_reports_handle_a_diagnostic_subset_without_entities(self):
        from cti.eval.report import export_report_csv, export_report_markdown

        summary = self.make_summary(gold=[["O", "O"]], relevant_mask=[False])
        diagnostic = summary["diagnostic_relevant_only"]
        self.assertEqual(diagnostic["relevant_sentences"], 0)
        self.assertEqual(diagnostic["evaluation"]["token_level"]["macro_f1"], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            export_report_csv(summary, Path(directory) / "report.csv")
            export_report_markdown(summary, Path(directory) / "report.md")

    def test_report_rejects_truncated_sentence_metadata(self):
        for override in ({"doc_ids": []}, {"sources": []}, {"relevant_mask": []}):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    self.make_summary(**override)


class CVAndBootstrapTests(unittest.TestCase):
    def test_report_grouped_splits_respect_boundaries(self):
        from cti.eval.cv import get_report_grouped_splits

        # 6 documents across 12 sentences
        doc_ids = ["docA", "docA", "docB", "docB", "docC", "docD", "docD", "docE", "docF", "docF", "docF", "docC"]
        splits = get_report_grouped_splits(doc_ids, n_splits=3, shuffle=True, seed=42)

        self.assertEqual(len(splits), 3)
        for train_idx, val_idx in splits:
            # Check non-empty
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(val_idx), 0)
            # Check partition
            self.assertEqual(len(train_idx) + len(val_idx), len(doc_ids))
            # Check zero group overlap
            train_docs = {doc_ids[i] for i in train_idx}
            val_docs = {doc_ids[i] for i in val_idx}
            self.assertEqual(len(train_docs & val_docs), 0)

    def test_cv_rejects_test_split_reports(self):
        from cti.eval.cv import get_locked_test_ids, get_report_grouped_splits

        test_ids = get_locked_test_ids()
        if test_ids:
            first_test_id = next(iter(test_ids))
            with self.assertRaises(PermissionError):
                get_report_grouped_splits([first_test_id, "docA", "docB"], n_splits=2)

    def test_cluster_bootstrap_produces_valid_intervals(self):
        from cti.eval.bootstrap import cluster_bootstrap_ci

        gold = [["B-ACTOR", "O"], ["O", "B-TOOL"], ["B-ACTOR", "O"]]
        pred = [["B-ACTOR", "O"], ["O", "O"], ["B-ACTOR", "O"]]
        docs = ["rep1", "rep2", "rep3"]

        def exact_f1(g, p):
            return exact_span_metrics(g, p)["f1"]

        pt, lo, hi = cluster_bootstrap_ci(gold, pred, docs, exact_f1, n_bootstraps=50, seed=1)
        self.assertGreaterEqual(pt, 0.0)
        self.assertLessEqual(pt, 1.0)
        self.assertLessEqual(lo, hi)
        self.assertTrue(lo <= pt <= hi or round(lo, 2) <= round(pt, 2) <= round(hi, 2))


if __name__ == "__main__":
    unittest.main()
