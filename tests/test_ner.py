import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from cti.core.base import Document, Entity
from cti.core.pipeline import Pipeline
from cti.extractors.ner import ActionNER, EntityNER
from cti.models.crf import CRFTagger
from cti.models.memm import MEMMTagger, TokenLogisticTagger
from cti.models.ner_artifact import load_ner_tagger, save_ner_tagger
from cti.models.ner_data import load_aligned_ner
from cti.output.merger import ResultMerger
from cti.preprocessing.tokenizer import tokenize


ROOT = Path(__file__).resolve().parents[1]
TEXTS = ["APT29 used Mimikatz.", "Lazarus used malware."]
TOKENS = [[piece[0] for piece in tokenize(text)] for text in TEXTS]
ENTITY_TAGS = [["B-THREAT_ACTOR", "O", "B-MALWARE_TOOL", "O"]] * 2
ACTION_TAGS = [["O", "B-ACTION", "I-ACTION", "O"]] * 2


def fitted(layer="entities", model="crf_tagger"):
    if model == "crf_tagger":
        tagger = CRFTagger(c1=0, c2=0.01, max_iterations=40)
    elif model == "memm_tagger":
        tagger = MEMMTagger(c=10, max_iter=100)
    else:
        tagger = TokenLogisticTagger(c=10, max_iter=100)
    tags = ENTITY_TAGS if layer == "entities" else ACTION_TAGS
    tagger.fit(TOKENS * 2, tags * 2)
    return tagger


class NERArtifactTests(unittest.TestCase):
    def test_all_classical_taggers_preserve_predictions_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("crf_tagger", "memm_tagger", "token_logreg"):
                with self.subTest(model=name):
                    tagger = fitted(model=name)
                    path = Path(directory) / f"{name}.joblib"
                    saved = save_ner_tagger(tagger, path, "entities", metadata={"split": "synthetic-train"})
                    loaded = load_ner_tagger(path, expected_layer="entities")
                    inputs = [TOKENS[0], [], TOKENS[1]]
                    self.assertEqual(loaded.tagger.predict(inputs), tagger.predict(inputs))
                    self.assertEqual(saved["artifact_sha256"], loaded.manifest["artifact_sha256"])
                    self.assertEqual(loaded.manifest["metadata"], {"split": "synthetic-train"})

    def test_unfitted_and_wrong_layer_models_cannot_be_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            with self.assertRaises(RuntimeError):
                save_ner_tagger(CRFTagger(), path, "entities")
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(ValueError, "outside"):
                save_ner_tagger(fitted("action"), path, "entities")
            self.assertFalse(path.exists())

    def test_layer_corruption_and_environment_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            tagger = fitted()
            save_ner_tagger(tagger, path, "entities")
            with self.assertRaisesRegex(ValueError, "Expected NER layer"):
                load_ner_tagger(path, expected_layer="action")
            with path.open("ab") as file:
                file.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_ner_tagger(path)
            save_ner_tagger(tagger, path, "entities")
            with patch("cti.models.ner_artifact._environment", return_value={}):
                with self.assertRaisesRegex(ValueError, "resources or runtime"):
                    load_ner_tagger(path)

    def test_manifest_version_and_neural_model_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            manifest = save_ner_tagger(fitted(), path, "entities")
            manifest_path = path.with_suffix(".manifest.json")
            for changes, message in (({"artifact_version": 999}, "version"), ({"model_name": "bert_ner"}, "deep learning")):
                manifest_path.write_text(json.dumps({**manifest, **changes}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_ner_tagger(path)


class NERPipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.entities = self.root / "entities.joblib"
        self.action = self.root / "action.joblib"
        save_ner_tagger(fitted(), self.entities, "entities")
        save_ner_tagger(fitted("action"), self.action, "action")

    def pipeline(self):
        config = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
        config["pipeline"][-1:-1] = [
            {"module": "ner_entities", "params": {"model_path": str(self.entities)}},
            {"module": "ner_action", "params": {"model_path": str(self.action)}},
        ]
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return Pipeline.from_config(path)

    def test_both_layers_emit_evidence_spans_and_keep_iocs(self):
        raw = "  APT29   used Mimikatz. It contacted 10.1.2[.]3. "
        document = self.pipeline().run(raw)
        findings = {(entity.label, entity.text) for entity in document.entities}
        self.assertIn(("THREAT_ACTOR", "APT29"), findings)
        self.assertIn(("MALWARE_TOOL", "Mimikatz"), findings)
        self.assertIn(("ACTION", "used Mimikatz"), findings)
        self.assertIn(("IP", "10.1.2.3"), findings)
        for entity in document.entities:
            self.assertEqual(document.text[entity.start:entity.end], entity.text)
            self.assertEqual(raw[entity.original_start:entity.original_end], entity.original_text)
            if entity.source.startswith("ner_"):
                self.assertIsNone(entity.confidence)
        self.assertEqual(document.metadata["ner"]["entities"]["offset_space"], "document_text")
        action = next(e for e in document.entities if e.label == "ACTION")
        self.assertEqual(action.original_text, "used Mimikatz")

    def test_repeated_inference_deduplicates_own_findings(self):
        document = self.pipeline().run(TEXTS[0])
        before = list(document.entities)
        merged_before = copy.deepcopy(document.findings)
        self.assertTrue(merged_before)
        EntityNER(self.entities).process(document)
        self.assertEqual(document.entities, before)
        self.assertEqual(document.findings, [])
        self.assertNotIn("merge", document.metadata)
        ResultMerger().process(document)
        self.assertEqual(document.findings, merged_before)

    def test_optional_relevance_gate_requires_predictions_and_keeps_other_entities(self):
        document = self.pipeline().run(TEXTS[0])
        document.entities = [Entity("APT29", "TEST", 0, 5, source="existing")]
        module = EntityNER(self.entities, use_relevance=True)
        with self.assertRaisesRegex(ValueError, "predicted boolean"):
            module.process(document)
        document.metadata["sentences"][0]["relevant"] = False
        module.process(document)
        self.assertEqual(len(document.entities), 1)
        self.assertEqual(document.metadata["ner"]["entities"]["sentences_tagged"], 0)

    def test_relevance_rerun_removes_stale_own_findings_only(self):
        document = self.pipeline().run(TEXTS[0])
        self.assertTrue(any(entity.source == "ner_entities" for entity in document.entities))
        document.metadata["sentences"][0]["relevant"] = False
        EntityNER(self.entities, use_relevance=True).process(document)
        self.assertFalse(any(entity.source == "ner_entities" for entity in document.entities))
        self.assertTrue(any(entity.source == "ner_action" for entity in document.entities))

    def test_invalid_prediction_lengths_and_labels_do_not_append_partial_results(self):
        module = EntityNER(self.entities)
        document = self.pipeline().run(TEXTS[0])
        before = copy.deepcopy(document.entities)
        for predictions in ([[]], [["B-ACTION", "O", "O", "O"]]):
            with self.subTest(predictions=predictions):
                with patch.object(module.artifact.tagger, "predict", return_value=predictions):
                    with self.assertRaises(ValueError):
                        module.process(document)
                self.assertEqual(document.entities, before)

    def test_sentence_and_token_offsets_are_validated(self):
        module = EntityNER(self.entities)
        document = self.pipeline().run(TEXTS[0])
        document.metadata["sentences"][0]["start"] = 1
        with self.assertRaisesRegex(ValueError, "Sentence offsets"):
            module.process(document)
        document = self.pipeline().run(TEXTS[0])
        with patch("cti.extractors.ner._tokenize", return_value=[("wrong", 0, 5)]):
            with self.assertRaisesRegex(ValueError, "Tokenizer offsets"):
                module.process(document)

    def test_unsupported_languages_and_empty_documents(self):
        module = EntityNER(self.entities)
        other = Document("हा हल्ला आहे", metadata={"language": "mr", "sentences": []})
        module.process(other)
        self.assertEqual(other.metadata["ner"]["entities"]["status"], "unsupported_language")
        empty = Document("", metadata={"language": "en", "sentences": []})
        module.process(empty)
        self.assertEqual(empty.entities, [])
        with self.assertRaisesRegex(ValueError, "sentence_splitter"):
            module.process(Document("malware"))


class NERDataAndTrainingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name in ("raw", "splits", "processed"):
            (self.root / name).mkdir()
        self.rows = []
        for split in ("train", "dev", "test"):
            doc_id = f"synthetic-{split}"
            (self.root / "splits" / f"{split}_report_ids.json").write_text(json.dumps([doc_id]), encoding="utf-8")
            records = []
            for index, text in enumerate(TEXTS):
                uid = f"{split}-{index}"
                self.rows.append({"uid": uid, "doc_id": doc_id, "sentence_text": text, "relevant": "yes", "source": "synthetic"})
                records.append({"uid": uid, "tokens": [list(piece) + [tag] for piece, tag in zip(tokenize(text), ENTITY_TAGS[index])]})
            if split != "test":
                (self.root / "processed" / f"bio_en_entities_{split}.json").write_text(json.dumps(records), encoding="utf-8")
        (self.root / "raw/SMAH-CTI.json").write_text(json.dumps(self.rows), encoding="utf-8")
        self.config = self.root / "training.yaml"
        self.config.write_text(yaml.safe_dump({
            "train_relevant_only": True,
            "layers": {"entities": {"model": "crf_tagger", "parameters": {"c1": 0, "c2": 0.01, "max_iterations": 30}}},
        }), encoding="utf-8")

    def train(self):
        from cti.train_ner import train_from_config

        return train_from_config(
            self.config, self.root / "raw", self.root / "splits", self.root / "processed",
            self.root / "models", self.root / "results", layers=("entities",),
        )

    def test_loader_joins_by_uid_instead_of_raw_row_order(self):
        rows = pd.DataFrame([row for row in self.rows if row["doc_id"] == "synthetic-train"][::-1])
        data = load_aligned_ner("entities", "train", rows, self.root / "processed")
        self.assertEqual(data.uids, ["train-0", "train-1"])
        self.assertEqual(data.tokens, TOKENS)
        with self.assertRaises(PermissionError):
            load_aligned_ner("entities", "test", rows, self.root / "processed")

    def test_missing_duplicate_and_invalid_bio_records_are_rejected(self):
        path = self.root / "processed/bio_en_entities_train.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        rows = pd.DataFrame([row for row in self.rows if row["doc_id"] == "synthetic-train"])
        damaged = copy.deepcopy(original)
        damaged[0]["tokens"][0][1] = 1
        for records in ([original[0]], original + [original[0]], damaged):
            with self.subTest(records=records):
                path.write_text(json.dumps(records), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_aligned_ner("entities", "train", rows, self.root / "processed")

    def test_train_dev_cli_pipeline_writes_versioned_results_and_reload_evidence(self):
        with patch.dict(os.environ, {"CTI_ALLOW_TEST": "1"}):
            reports = self.train()
        self.assertEqual(reports["entities"]["metrics_version"], "2")
        self.assertTrue(reports["entities"]["reload_predictions_identical"])
        self.assertFalse(reports["entities"]["artifact"]["metadata"]["test_split_used"])
        self.assertTrue((self.root / "results/ner_current_entities.predictions.json").is_file())
        artifact = load_ner_tagger(self.root / "models/ner_entities.joblib")
        self.assertEqual(artifact.manifest["metadata"]["train_report_ids"], ["synthetic-train"])

    def test_neural_configs_and_split_leakage_are_rejected(self):
        config = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        config["layers"]["entities"]["model"] = "bert_ner"
        self.config.write_text(yaml.safe_dump(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "deep learning"):
            self.train()
        config["layers"]["entities"]["model"] = "crf_tagger"
        self.config.write_text(yaml.safe_dump(config), encoding="utf-8")
        (self.root / "splits/test_report_ids.json").write_text(json.dumps(["synthetic-train"]), encoding="utf-8")
        with self.assertRaises(PermissionError):
            self.train()
        self.assertFalse((self.root / "models").exists())

    def test_unknown_hyperparameters_are_not_silently_ignored(self):
        config = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        config["layers"]["entities"]["parameters"]["typo_parameter"] = 10
        self.config.write_text(yaml.safe_dump(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsupported.*parameters"):
            self.train()
        self.assertFalse((self.root / "models").exists())


if __name__ == "__main__":
    unittest.main()
