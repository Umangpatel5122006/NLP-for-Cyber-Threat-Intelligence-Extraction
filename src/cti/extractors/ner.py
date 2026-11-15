"""Saved classical BIO tagging over document sentences with explicit offsets."""

from cti.core.base import BaseModule, Document, Entity
from cti.core.registry import register
from cti.models.base import validate_training_data
from cti.models.ner_artifact import load_ner_tagger


def _tokenize(text):
    # Loading an unused document module must not import the tokenizer stack.
    from cti.preprocessing.tokenizer import tokenize

    return tokenize(text, lang="en")


class _NERModule(BaseModule):
    layer = None

    def __init__(self, model_path, use_relevance=False):
        if not isinstance(use_relevance, bool):
            raise TypeError("use_relevance must be a boolean.")
        self.artifact = load_ner_tagger(model_path, expected_layer=self.layer)
        self.use_relevance = use_relevance

    def _store(self, doc, details, findings=()):
        # A rerun refreshes this module's output without deleting IOC/other-layer evidence.
        findings = [doc.project_entity(finding) for finding in findings]
        remaining = [entity for entity in doc.entities if entity.source != self.name]
        seen = {(entity.start, entity.end, entity.label, entity.source) for entity in remaining}
        for finding in findings:
            key = (finding.start, finding.end, finding.label, finding.source)
            if key not in seen:
                remaining.append(finding)
                seen.add(key)
        doc.entities[:] = sorted(remaining, key=lambda entity: (entity.start, entity.end, entity.label, entity.source))
        doc.invalidate_findings()
        doc.metadata.setdefault("ner", {})[self.layer] = details
        return doc

    def process(self, doc: Document) -> Document:
        if "language" not in doc.metadata or "sentences" not in doc.metadata:
            raise ValueError("Run language_detector and sentence_splitter before NER.")
        sentences = doc.metadata["sentences"]
        if not isinstance(sentences, list):
            raise ValueError("sentences must be a list.")
        if "ner" in doc.metadata and not isinstance(doc.metadata["ner"], dict):
            raise ValueError("NER metadata must be a mapping.")
        details = {
            "status": "ok", "layer": self.layer,
            "model_name": self.artifact.manifest["model_name"],
            "artifact_sha256": self.artifact.manifest["artifact_sha256"],
            "offset_space": "document_text", "confidence_kind": "unavailable",
            "use_relevance": self.use_relevance, "sentences_tagged": 0,
        }
        if doc.metadata["language"] != "en":
            details["status"] = "unsupported_language"
            return self._store(doc, details)

        selected, tokenized = [], []
        for sentence in sentences:
            if not isinstance(sentence, dict):
                raise ValueError("Each sentence must be a mapping with text/start/end.")
            text, start, end = (sentence.get(key) for key in ("text", "start", "end"))
            if (
                not isinstance(text, str) or type(start) is not int or type(end) is not int
                or not 0 <= start <= end <= len(doc.text) or doc.text[start:end] != text
            ):
                raise ValueError("Sentence offsets must match the current document text.")
            if self.use_relevance:
                if type(sentence.get("relevant")) is not bool:
                    raise ValueError("Relevance gating requires predicted boolean sentence decisions.")
                if not sentence["relevant"]:
                    continue
            pieces = _tokenize(text)
            previous_end = 0
            for token, token_start, token_end in pieces:
                if not (previous_end <= token_start < token_end <= len(text)) or text[token_start:token_end] != token:
                    raise ValueError("Tokenizer offsets must match sentence text.")
                previous_end = token_end
            selected.append(sentence)
            tokenized.append(pieces)
        tokens = [[piece[0] for piece in pieces] for pieces in tokenized]
        if not tokens:
            return self._store(doc, details)
        predictions = self.artifact.tagger.predict(tokens)
        validate_training_data(tokens, predictions)
        findings = []
        for sentence, pieces, tags in zip(selected, tokenized, predictions):
            span_start = span_end = label = None
            for index in range(len(tags) + 1):
                tag = tags[index] if index < len(tags) else "O"
                if tag != "O" and tag[2:] not in self.artifact.manifest["labels"]:
                    raise ValueError(f"Prediction {tag!r} is outside the {self.layer} label layer.")
                if tag == "O" or tag.startswith("B-"):
                    if label is not None:
                        start, end = sentence["start"] + span_start, sentence["start"] + span_end
                        findings.append(Entity(
                            text=doc.text[start:end], label=label, start=start, end=end,
                            source=self.name, confidence=None,
                        ))
                    span_start = span_end = label = None
                if tag.startswith("B-"):
                    label, span_start = tag[2:], pieces[index][1]
                if tag != "O":
                    span_end = pieces[index][2]
        # Commit findings only after all predictions pass validation.
        details["sentences_tagged"] = len(selected)
        return self._store(doc, details, findings)


@register("ner_entities")
class EntityNER(_NERModule):
    name = "ner_entities"
    layer = "entities"


@register("ner_action")
class ActionNER(_NERModule):
    name = "ner_action"
    layer = "action"
