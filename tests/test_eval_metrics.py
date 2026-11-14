import unittest

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
