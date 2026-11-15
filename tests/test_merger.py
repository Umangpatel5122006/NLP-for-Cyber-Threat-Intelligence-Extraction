import copy
import json
import unittest
from dataclasses import asdict
from pathlib import Path

from cti.core.base import Document, Entity
from cti.core.pipeline import Pipeline
from cti.core.registry import create
from cti.extractors.ioc_finder import IOCFinder
from cti.extractors.ioc_regex import IOCRegex
from cti.output.merger import ResultMerger
from cti.preprocessing.cleaner import TextCleaner
from cti.preprocessing.defanger import Defanger


ROOT = Path(__file__).resolve().parents[1]


def document(raw):
    return Defanger().process(TextCleaner().process(Document(raw)))


def observe(doc, text, label, source="synthetic", confidence=None, start=None):
    if start is None:
        start = doc.text.index(text)
    observation = Entity(text, label, start, start + len(text), source, confidence)
    doc.entities.append(observation)
    return observation


class MergerTests(unittest.TestCase):
    def merge(self, doc):
        return ResultMerger().process(doc)

    def test_empty_input_and_registry_creation(self):
        doc = create("result_merger").process(Document(""))
        self.assertEqual(doc.findings, [])
        self.assertEqual(doc.overlaps, [])
        self.assertEqual(doc.metadata["merge"]["finding_count"], 0)

    def test_exact_duplicates_are_removed_from_the_view_without_changing_raw_observations(self):
        doc = document(" 185.22.10[.]5 ")
        for _ in range(3):
            observe(doc, "185.22.10.5", "IP", "ioc_regex", 1.0)
        before = copy.deepcopy(doc.entities)
        self.merge(doc)
        self.assertEqual(doc.entities, before)
        self.assertEqual(len(doc.findings), 1)
        self.assertEqual(len(doc.findings[0].evidence), 1)
        self.assertEqual(doc.findings[0].original_text, "185.22.10[.]5")
        self.assertEqual(doc.metadata["merge"]["duplicate_evidence_count"], 2)

    def test_multiple_sources_and_scores_remain_separate_evidence(self):
        doc = document("185.22.10.5")
        observe(doc, doc.text, "IP", "ioc_finder", 1.0)
        observe(doc, doc.text, "IP", "ioc_regex", None)
        observe(doc, doc.text, "IP", "ioc_regex", 0.0)
        self.merge(doc)
        finding = doc.findings[0]
        self.assertEqual(finding.sources, ["ioc_finder", "ioc_regex"])
        self.assertEqual([e.confidence for e in finding.evidence], [1.0, None, 0.0])
        self.assertFalse(hasattr(finding, "confidence"))
        self.assertEqual(doc.metadata["merge"]["confidence_policy"], "per_source_only")

    def test_specific_ioc_carries_exact_span_infrastructure_evidence(self):
        doc = document(" 185.22.10[.]5 ")
        observe(doc, "185.22.10.5", "INFRASTRUCTURE", "ner_entities")
        observe(doc, "185.22.10.5", "IP", "ioc_finder", 1.0)
        self.merge(doc)
        finding = doc.findings[0]
        self.assertEqual(len(doc.findings), 1)
        self.assertEqual(finding.label, "IP")
        self.assertEqual(finding.observed_labels, ["INFRASTRUCTURE", "IP"])
        self.assertEqual(finding.sources, ["ioc_finder", "ner_entities"])
        self.assertEqual(len(finding.evidence), 2)
        self.assertEqual(doc.metadata["merge"]["absorbed_infrastructure_count"], 1)

    def test_broad_infrastructure_spans_are_retained(self):
        doc = document("server 185.22.10.5")
        observe(doc, doc.text, "INFRASTRUCTURE", "ner_entities")
        observe(doc, "185.22.10.5", "IP", "ioc_finder", 1.0)
        self.merge(doc)
        self.assertEqual(len(doc.findings), 2)
        self.assertEqual(doc.overlaps[0].kind, "contained_overlap")
        self.assertTrue(all(f.role == "primary" for f in doc.findings))

    def test_ambiguous_specific_types_do_not_absorb_infrastructure(self):
        doc = document("185.22.10.5")
        for label in ("IP", "DOMAIN", "INFRASTRUCTURE"):
            observe(doc, doc.text, label)
        self.merge(doc)
        self.assertEqual(len(doc.findings), 3)
        self.assertEqual(len(doc.overlaps), 3)
        self.assertTrue(all(edge.kind == "label_conflict" for edge in doc.overlaps))

    def test_invalid_ioc_syntax_remains_visible_without_overriding_infrastructure(self):
        doc = document("not-an-ip")
        for label in ("IP", "INFRASTRUCTURE"):
            observe(doc, doc.text, label)
        self.merge(doc)
        self.assertEqual(len(doc.findings), 2)
        ip = next(f for f in doc.findings if f.label == "IP")
        self.assertIsNone(ip.canonical_value)
        self.assertEqual(ip.canonical_status, "unavailable")

    def test_canonical_values_do_not_change_evidence_text_or_positions(self):
        pairs = [
            ("DOMAIN", "EVIL.Example.COM.", "evil.example.com"),
            ("IP", "2001:0DB8:0:0:0:0:0:1", "2001:db8::1"),
            ("HASH", "ABCDEF0123456789ABCDEF0123456789", "abcdef0123456789abcdef0123456789"),
            ("CVE", "cve-2024-12345", "CVE-2024-12345"),
        ]
        for label, text, expected in pairs:
            with self.subTest(label=label):
                doc = document("  " + text + "  ")
                observe(doc, text, label)
                self.merge(doc)
                finding = doc.findings[0]
                self.assertEqual(finding.canonical_value, expected)
                self.assertEqual(finding.canonical_status, "available")
                self.assertEqual(finding.text, text)
                self.assertEqual(finding.original_text, text)
                self.assertEqual((finding.original_start, finding.original_end), (2, 2 + len(text)))

    def test_repeated_canonical_values_remain_separate_occurrences(self):
        doc = document("EVIL.COM then evil.com")
        observe(doc, "EVIL.COM", "DOMAIN")
        observe(doc, "evil.com", "DOMAIN")
        self.merge(doc)
        self.assertEqual([f.canonical_value for f in doc.findings], ["evil.com", "evil.com"])
        self.assertNotEqual(doc.findings[0].id, doc.findings[1].id)
        self.assertEqual(doc.overlaps, [])

    def test_url_normalization_preserves_userinfo_port_path_query_and_fragment(self):
        text = "HTTPS://User:SeCrEt@EXAMPLE.COM:443/PaTh?Token=AbC#Section"
        doc = document(text)
        observe(doc, text, "URL")
        self.merge(doc)
        self.assertEqual(doc.findings[0].canonical_value, "https://User:SeCrEt@example.com:443/PaTh?Token=AbC#Section")
        # Empty query/fragment delimiters are also retained.
        doc = document("HTTPS://EXAMPLE.COM/?#")
        observe(doc, doc.text, "URL")
        self.merge(doc)
        self.assertEqual(doc.findings[0].canonical_value, "https://example.com/?#")

    def test_url_host_domain_is_a_linked_component(self):
        doc = document("hxxps://EVIL[.]Example[.]COM/A")
        observe(doc, doc.text, "URL", "ioc_finder", 1.0)
        observe(doc, "EVIL.Example.COM", "DOMAIN", "ioc_finder", 1.0)
        self.merge(doc)
        url = next(f for f in doc.findings if f.label == "URL")
        domain = next(f for f in doc.findings if f.label == "DOMAIN")
        self.assertEqual(url.component_ids, [domain.id])
        self.assertEqual(domain.parent_ids, [url.id])
        self.assertEqual(domain.role, "component")
        self.assertEqual(domain.original_text, "EVIL[.]Example[.]COM")
        self.assertEqual(doc.overlaps[0].kind, "ioc_host_component")
        self.assertEqual(doc.metadata["merge"]["primary_finding_count"], 1)

    def test_domains_in_url_paths_and_queries_stay_primary(self):
        text = "https://OTHER.EXAMPLE.ORG/evil.example.com?next=evil.example.com"
        doc = document(text)
        observe(doc, text, "URL")
        observe(doc, "OTHER.EXAMPLE.ORG", "DOMAIN")
        observe(doc, "evil.example.com", "DOMAIN")
        observe(doc, "evil.example.com", "DOMAIN", start=text.rindex("evil.example.com"))
        self.merge(doc)
        domains = [f for f in doc.findings if f.text == "evil.example.com"]
        self.assertEqual(len(domains), 2)
        self.assertTrue(all(f.role == "primary" and not f.parent_ids for f in domains))
        self.assertEqual(sum(e.kind == "ioc_host_component" for e in doc.overlaps), 1)

    def test_bracketed_ipv6_host_component_excludes_brackets(self):
        doc = document("HTTPS://[2001:0DB8:0:0::1]:8443/A?x=Z")
        observe(doc, doc.text, "URL")
        observe(doc, "2001:0DB8:0:0::1", "IP")
        self.merge(doc)
        url = next(f for f in doc.findings if f.label == "URL")
        ip = next(f for f in doc.findings if f.label == "IP")
        self.assertEqual(url.canonical_value, "https://[2001:db8::1]:8443/A?x=Z")
        self.assertEqual(ip.parent_ids, [url.id])
        self.assertEqual(ip.canonical_value, "2001:db8::1")

    def test_email_local_part_is_preserved_and_only_its_domain_is_a_component(self):
        doc = document("Admin.Name[at]EXAMPLE[.]COM")
        observe(doc, doc.text, "EMAIL")
        observe(doc, "Admin.Name", "DOMAIN")
        observe(doc, "EXAMPLE.COM", "DOMAIN")
        self.merge(doc)
        email = next(f for f in doc.findings if f.label == "EMAIL")
        local = next(f for f in doc.findings if f.text == "Admin.Name")
        host = next(f for f in doc.findings if f.text == "EXAMPLE.COM")
        self.assertEqual(email.canonical_value, "Admin.Name@example.com")
        self.assertEqual(local.role, "primary")
        self.assertEqual(host.parent_ids, [email.id])

    def test_malformed_or_unsupported_values_retain_raw_evidence(self):
        pairs = [
            ("URL", "https://[::1]junk/A"), ("URL", "https://example.com:invalid/A"),
            ("URL", "relative/path"), ("URL", "https://[broken/A"),
            ("DOMAIN", "example..com"), ("DOMAIN", "caf\u00e9.example"),
            ("EMAIL", "@example.com"), ("HASH", "bad-hash"), ("CVE", "CVE-24-12"),
        ]
        for label, text in pairs:
            with self.subTest(label=label, text=text):
                doc = document(text)
                observe(doc, doc.text, label)
                self.merge(doc)
                self.assertEqual(doc.findings[0].canonical_status, "unavailable")
                self.assertEqual(doc.findings[0].original_text, text)

    def test_actions_and_their_nested_entities_remain_primary(self):
        doc = document("APT29 used Mimikatz")
        observe(doc, "used Mimikatz", "ACTION", "ner_action")
        observe(doc, "Mimikatz", "MALWARE_TOOL", "ner_entities")
        self.merge(doc)
        self.assertTrue(all(f.role == "primary" for f in doc.findings))
        self.assertEqual(doc.overlaps[0].kind, "action_overlap")
        self.assertTrue(all(f.canonical_status == "not_applicable" for f in doc.findings))

    def test_conflicting_exact_labels_are_retained(self):
        doc = document("Example")
        for label in ("THREAT_ACTOR", "MALWARE_TOOL"):
            observe(doc, doc.text, label, "ner_entities")
        self.merge(doc)
        self.assertEqual(len(doc.findings), 2)
        self.assertEqual(doc.overlaps[0].kind, "label_conflict")

    def test_crossing_spans_are_retained_and_identified(self):
        doc = document("alpha beta gamma")
        observe(doc, "alpha beta", "INFRASTRUCTURE")
        observe(doc, "beta gamma", "MALWARE_TOOL")
        self.merge(doc)
        self.assertEqual(doc.overlaps[0].kind, "crossing_overlap")

    def test_view_is_deterministic_across_input_order_and_reruns(self):
        doc = document("https://EVIL.COM used Mimikatz")
        observe(doc, "https://EVIL.COM", "URL")
        observe(doc, "EVIL.COM", "DOMAIN", "source-b", 1.0)
        observe(doc, "EVIL.COM", "DOMAIN", "source-a", None)
        observe(doc, "used Mimikatz", "ACTION")
        observe(doc, "Mimikatz", "MALWARE_TOOL")
        other = copy.deepcopy(doc)
        other.entities.reverse()
        self.merge(doc)
        self.merge(other)
        self.assertEqual(doc.findings, other.findings)
        self.assertEqual(doc.overlaps, other.overlaps)
        self.assertEqual(doc.metadata["merge"], other.metadata["merge"])
        before = copy.deepcopy(asdict(doc))
        self.merge(doc)
        self.assertEqual(asdict(doc), before)

    def test_invalid_observations_do_not_replace_an_existing_view(self):
        invalid = [
            Entity("wrong", "IP", 0, 10), Entity("185.22.10.5", "", 0, 11),
            Entity("185.22.10.5", "IP", 0, 11, source=None),
        ]
        invalid += [Entity("185.22.10.5", "IP", 0, 11, confidence=score)
                    for score in (float("nan"), float("inf"), -0.1, 1.1, True, "1", 10**999)]
        for bad in invalid:
            with self.subTest(bad=bad):
                doc = document("185.22.10.5")
                observe(doc, doc.text, "IP")
                self.merge(doc)
                old_findings, old_metadata = copy.deepcopy(doc.findings), copy.deepcopy(doc.metadata)
                doc.entities.append(bad)
                with self.assertRaises((ValueError, TypeError)):
                    self.merge(doc)
                self.assertEqual(doc.findings, old_findings)
                self.assertEqual(doc.metadata, old_metadata)
                self.assertIsNone(bad.original_start)

    def test_evidence_copies_recompute_original_fields_without_mutating_observations(self):
        doc = document(" 185.22.10[.]5 ")
        original = observe(doc, doc.text, "IP")
        original.original_text, original.original_start, original.original_end = "stale", 99, 100
        self.merge(doc)
        evidence = doc.findings[0].evidence[0]
        self.assertEqual(evidence.original_text, "185.22.10[.]5")
        self.assertEqual(original.original_text, "stale")
        evidence.source = "changed-view"
        self.assertEqual(original.source, "synthetic")

    def test_detector_reruns_invalidate_the_view_until_rebuilt(self):
        for module in (IOCRegex(), IOCFinder()):
            with self.subTest(module=module.name):
                doc = module.process(document("185.22.10[.]5"))
                self.merge(doc)
                self.assertTrue(doc.findings)
                module.process(doc)
                self.assertEqual(doc.findings, [])
                self.assertEqual(doc.overlaps, [])
                self.assertNotIn("merge", doc.metadata)
                self.merge(doc)
                self.assertEqual(len(doc.findings), 1)

    def test_merger_order_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "after all extraction"):
            Pipeline([ResultMerger(), IOCRegex()])
        with self.assertRaises(ValueError):
            Pipeline([ResultMerger(), ResultMerger()])

    def test_configured_pipeline_exports_resolvable_evidence_and_relationships(self):
        raw = "  Contact hxxps://EVIL[.]example[.]com/A or Admin[at]EVIL[.]example[.]com. "
        doc = Pipeline.from_config(ROOT / "configs/config.yaml").run(raw)
        payload = json.loads(json.dumps(asdict(doc), allow_nan=False))
        self.assertTrue(payload["findings"])
        identifiers = {f["id"] for f in payload["findings"]}
        for finding in payload["findings"]:
            self.assertEqual(raw[finding["original_start"]:finding["original_end"]], finding["original_text"])
            self.assertTrue(set(finding["parent_ids"] + finding["component_ids"]) <= identifiers)
            self.assertTrue(finding["evidence"])
        for edge in payload["overlaps"]:
            self.assertTrue({edge["left_id"], edge["right_id"]} <= identifiers)
        self.assertEqual(len(doc.entities), doc.metadata["merge"]["raw_entity_count"])

    def test_real_ioc_alignment_retains_mixed_case_and_distinct_url_paths(self):
        raw = "HTTPS://EVIL.example.com/A https://evil.EXAMPLE.com/a Admin@EVIL.example.com"
        doc = Pipeline.from_config(ROOT / "configs/config.yaml").run(raw)
        urls = [f for f in doc.findings if f.label == "URL"]
        self.assertEqual([f.text for f in urls], ["HTTPS://EVIL.example.com/A", "https://evil.EXAMPLE.com/a"])
        self.assertEqual([f.canonical_value for f in urls], ["https://evil.example.com/A", "https://evil.example.com/a"])
        email = next(f for f in doc.findings if f.label == "EMAIL")
        self.assertEqual(email.text, "Admin@EVIL.example.com")
        self.assertEqual(email.canonical_value, "Admin@evil.example.com")


if __name__ == "__main__":
    unittest.main()
