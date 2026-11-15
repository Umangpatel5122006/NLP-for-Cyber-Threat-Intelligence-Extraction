"""Synthetic diagnostics checks, including exhaustive small matching cases."""

import itertools
import unittest

from cti.eval.action_diagnostics import (action_spans, compare_gold, diagnose, duplicate_annotations,
                                        illegal_inside_count, length_bucket, pair_spans)


class ActionDiagnosticsTests(unittest.TestCase):
    def test_exact_boundary_missed_and_spurious_counts_conserve_both_sides(self):
        result = diagnose([["B-ACTION", "I-ACTION", "O", "B-ACTION", "O", "O"]],
                          [["B-ACTION", "O", "O", "O", "O", "B-ACTION"]], [True])
        counts = result["taxonomy_exact_first"]
        self.assertEqual((counts["boundary_pairs"], counts["missed"], counts["spurious"]), (1, 1, 1))
        self.assertEqual(result["gold_any_overlap_status"], {"exact": 0, "overlap_only": 1, "no_overlap": 1})
        self.assertEqual(result["boundary_pair_types"]["prediction_inside_gold"], 1)

    def test_merged_prediction_cannot_claim_two_paired_gold_actions(self):
        result = diagnose([["B-ACTION", "O", "B-ACTION"]], [["B-ACTION", "I-ACTION", "I-ACTION"]], [True])
        self.assertEqual(result["taxonomy_exact_first"]["boundary_pairs"], 1)
        self.assertEqual(result["taxonomy_exact_first"]["missed"], 1)
        self.assertEqual(result["gold_any_overlap_status"]["overlap_only"], 2)
        self.assertEqual(result["taxonomy_exact_first"]["predicted_with_multiple_gold_overlaps"], 1)

    def test_split_prediction_cannot_claim_two_paired_true_positives(self):
        result = diagnose([["B-ACTION", "I-ACTION", "I-ACTION"]], [["B-ACTION", "O", "B-ACTION"]], [True])
        self.assertEqual(result["taxonomy_exact_first"]["boundary_pairs"], 1)
        self.assertEqual(result["taxonomy_exact_first"]["spurious"], 1)
        self.assertEqual(result["taxonomy_exact_first"]["gold_with_multiple_predicted_overlaps"], 1)

    def test_exact_matches_are_reserved_and_adjacent_spans_do_not_overlap(self):
        result = pair_spans([("ACTION", 0, 1), ("ACTION", 2, 3)], [("ACTION", 0, 1), ("ACTION", 3, 4)])
        self.assertEqual(result["exact"], [("ACTION", 0, 1)])
        self.assertEqual(result["boundary_pairs"], [])
        self.assertEqual(len(result["missed"]), 1)

    def test_strict_inside_start_is_counted_without_repairing_source_labels(self):
        result = diagnose([["I-ACTION", "O"]], [["B-ACTION", "O"]], [True])
        self.assertEqual(result["exact_strict"]["gold_spans"], 0)
        self.assertEqual(result["illegal_inside_starts"], {"gold": 1, "predicted": 0})
        self.assertEqual(illegal_inside_count(["B-ACTION", "I-ACTION", "O", "I-ACTION"]), 1)

    def test_gold_transition_counts_have_one_entry_per_gold_span(self):
        gold = [["B-ACTION", "I-ACTION", "O", "B-ACTION"]]
        before = [["B-ACTION", "I-ACTION", "O", "B-ACTION"]]
        after = [["B-ACTION", "O", "O", "O"]]
        transitions = compare_gold(gold, before, after)
        self.assertEqual(transitions["exact->overlap_only"], 1)
        self.assertEqual(transitions["exact->no_overlap"], 1)
        self.assertEqual(sum(transitions.values()), 2)

    def test_gold_length_buckets_and_background_counts_are_separate(self):
        gold = [["B-ACTION"] + ["I-ACTION"] * 19, ["O"]]
        predictions = [gold[0][:], ["B-ACTION"]]
        result = diagnose(gold, predictions, [True, False])
        self.assertEqual(result["gold_status_by_token_length"]["20+"]["exact"], 1)
        self.assertEqual(result["taxonomy_exact_first"]["background_predicted_spans"], 1)
        self.assertEqual([length_bucket(i) for i in (1, 2, 4, 5, 9, 10, 19, 20)],
                         ["1", "2-4", "2-4", "5-9", "5-9", "10-19", "10-19", "20+"])

    def test_duplicate_audit_distinguishes_tag_conflicts_and_relevance_conflicts(self):
        result = duplicate_annotations([["Used"], ["Used"], ["Used"], ["used"]],
                                       [["B-ACTION"], ["O"], ["B-ACTION"], ["O"]],
                                       [True, True, False, False], ["a", "b", "c", "d"])
        self.assertEqual(result["duplicate_groups"], 1)
        self.assertEqual(result["rows_in_duplicate_groups"], 3)
        self.assertEqual(result["conflicting_action_tag_groups"], 1)
        self.assertEqual(result["conflicting_relevance_groups"], 1)

    def test_rejects_alignment_errors_and_other_entity_layers(self):
        for gold, pred, relevant in (([["O"]], [], [True]), ([["O"]], [[]], [True]),
                                      ([["B-MALWARE_TOOL"]], [["O"]], [True]), ([["O"]], [["O"]], [1])):
            with self.subTest(gold=gold), self.assertRaises(ValueError):
                diagnose(gold, pred, relevant)

    def test_empty_sequences_have_zero_counts_without_division_errors(self):
        self.assertEqual(diagnose([[]], [[]], [False])["exact_strict"]["f1"], 0)
        self.assertEqual(diagnose([], [], [])["gold_lengths"]["count"], 0)

    def test_residual_greedy_pairs_match_exhaustive_maximum_for_small_legal_sequences(self):
        legal = [action_spans(list(tags)) for tags in itertools.product(("O", "B-ACTION", "I-ACTION"), repeat=4)
                 if not illegal_inside_count(list(tags))]
        def maximum(gs, ps, index=0, used=frozenset()):
            if index == len(gs):
                return 0
            best = maximum(gs, ps, index+1, used)
            for n, p in enumerate(ps):
                if n not in used and max(gs[index][1], p[1]) < min(gs[index][2], p[2]):
                    best = max(best, 1 + maximum(gs, ps, index+1, used | {n}))
            return best
        for gs in legal:
            for ps in legal:
                paired = pair_spans(gs, ps)
                exact = set(gs) & set(ps)
                expected = maximum([g for g in gs if g not in exact], [p for p in ps if p not in exact])
                self.assertEqual(len(paired["boundary_pairs"]), expected)
                self.assertEqual(len(gs), len(paired["exact"]) + expected + len(paired["missed"]))
                self.assertEqual(len(ps), len(paired["exact"]) + expected + len(paired["spurious"]))


if __name__ == "__main__":
    unittest.main()
