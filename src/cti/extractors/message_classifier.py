"""Optional short-message classical classifier; rules/IOCs survive missing models."""

from pathlib import Path

from cti.core.base import BaseModule
from cti.core.registry import register


@register("message_classifier")
class MessageSpamClassifier(BaseModule):
    name = "message_classifier"

    def __init__(self, model_path="models/message_spam.joblib", max_characters=1000):
        if type(max_characters) is not int or max_characters < 1:
            raise ValueError("Message length limit must be a positive integer.")
        self.path, self.max_characters = Path(model_path), max_characters
        self.artifact, self._identity, self._failure = None, None, None

    def readiness(self, retry=False):
        paths = (self.path, self.path.with_suffix(".manifest.json"))
        try:
            identity = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        except OSError:
            self.artifact, self._identity, self._failure = None, None, "models_missing"
            return {"ready": False, "code": self._failure, "message": "SMS classifier is missing. Rules and IOC extraction remain available.",
                    "setup": "python -m cti.train_message --download"}
        if identity != self._identity or retry and self.artifact is None:
            self._identity, self.artifact, self._failure = identity, None, None
            try:
                from cti.models.message_v2 import load_message_classifier
                self.artifact = load_message_classifier(self.path)
            except (ImportError, OSError):
                self._failure = "runtime_unavailable"
            except Exception:
                self._failure = "models_incompatible"
        if self.artifact is None:
            return {"ready": False, "code": self._failure, "message": "SMS classifier cannot load in this runtime. Rules remain available.",
                    "setup": "Use the prepared WSL runtime or retrain with python -m cti.train_message --download."}
        return {"ready": True, "profile": self.artifact.manifest["profile"],
                "training_protocol": self.artifact.manifest["metadata"].get("training_protocol", "uci_sms_v1"),
                "artifact_sha256": self.artifact.manifest["artifact_sha256"], "max_characters": self.max_characters,
                "dev_metrics": self.artifact.manifest["metadata"].get("dev_metrics", {})}

    def process(self, doc):
        doc.invalidate_findings()
        if len(doc.original_text) > self.max_characters:
            result = {"status": "not_applicable", "reason": f"Input exceeds the short-message classifier's {self.max_characters}-character scope."}
        elif doc.metadata.get("language") != "en":
            result = {"status": "not_applicable", "reason": "SMS classifier supports English messages."}
        elif not doc.original_text.strip():
            result = {"status": "not_applicable", "reason": "Message is empty."}
        else:
            readiness = self.readiness()
            if not readiness["ready"]:
                result = {"status": "unavailable", **readiness}
            else:
                try:
                    result = self.artifact.predict_one(doc.original_text)
                    result["training_protocol"] = self.artifact.manifest["metadata"].get("training_protocol", "uci_sms_v1")
                    result["training_sources"] = self.artifact.manifest["metadata"].get("training_sources", ["uci"])
                except Exception:
                    result = {"status": "unavailable", "code": "prediction_failed",
                              "message": "SMS prediction failed for this input. Rule/IOC evidence remains available."}
        doc.metadata["message_classifier"] = result
        return doc
