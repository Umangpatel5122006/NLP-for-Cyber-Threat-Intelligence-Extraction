"""Human review findings combine extraction and message evidence without relabeling."""

import math
from dataclasses import asdict


def build_review_findings(doc, extraction_findings=None):
    rows = []
    for finding in doc.findings if extraction_findings is None else extraction_findings:
        rows.append({**asdict(finding), "kind": "extraction", "extraction_id": finding.id,
                     "title": finding.text, "status": "extracted", "description": "Extracted source occurrence; presence alone does not establish maliciousness."})
    risk = doc.metadata.get("message_risk", {})
    if risk.get("status") in ("suspicious", "review_context"):
        evidence = []
        for signal in risk.get("signals", []):
            start, end = signal.get("original_start"), signal.get("original_end")
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(doc.original_text)
                    or doc.original_text[start:end] != signal.get("original_text")):
                raise ValueError("Message-risk evidence must match original offsets.")
            evidence.append({**signal, "source": "message_risk", "label": "SOCIAL_ENGINEERING", "confidence": None})
        if not evidence:
            raise ValueError("A positive message-rule finding must have source evidence.")
        start = min(e["original_start"] for e in evidence)
        end = max(e["original_end"] for e in evidence)
        rows.append({"id": "review-message-rules", "kind": "message_rule", "label": "SOCIAL_ENGINEERING",
                     "title": "Suspicious message patterns" if risk["status"] == "suspicious" else "Scam patterns in review context",
                     "text": doc.original_text[start:end], "original_text": doc.original_text[start:end],
                     "original_start": start, "original_end": end, "sources": ["message_risk"],
                     "evidence": evidence, "role": "primary", "status": risk["status"], "description": risk["summary"],
                     "matched_rules": risk["matched_rules"], "parent_ids": [], "component_ids": []})
    model = doc.metadata.get("message_classifier", {})
    if model.get("status") == "ok" and model.get("label") == "spam":
        score = model.get("decision_margin")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("SMS model finding requires a finite decision margin.")
        rows.append({"id": "review-message-model", "kind": "message_model", "label": "SMS_SPAM",
                     "title": "SMS spam predicted", "text": doc.original_text, "original_text": doc.original_text,
                     "original_start": 0, "original_end": len(doc.original_text), "sources": ["message_classifier"],
                     "evidence": [{"source": "message_classifier", "label": "SMS_SPAM", "confidence": None,
                                   "original_start": 0, "original_end": len(doc.original_text), "original_text": doc.original_text}],
                     "role": "primary", "status": "review_context" if risk.get("status") == "review_context" else "predicted_spam", "description": model["limitations"],
                     "decision_margin": score, "supporting_features": model["supporting_features"],
                     "profile": model["profile"], "parent_ids": [], "component_ids": []})
    return rows


def message_assessment(doc):
    risk, model = doc.metadata.get("message_risk", {}), doc.metadata.get("message_classifier", {})
    rule_positive = risk.get("status") == "suspicious"
    model_positive = model.get("status") == "ok" and model.get("label") == "spam"
    if risk.get("status") == "review_context":
        status, summary = "review_context", risk["summary"]
    elif rule_positive:
        status, summary = "suspicious", risk["summary"]
    elif model_positive:
        status, summary = "suspicious", "The classical SMS classifier predicts spam. Review the message; this is not verified fraud."
    else:
        status, summary = "no_pattern_detected", "No supported message-risk detection. This does not establish safety; inspect model coverage and extracted evidence."
    return {"status": status, "summary": summary, "rule_status": risk.get("status", "unavailable"),
            "model_status": model.get("status", "unavailable"), "model_label": model.get("label"),
            "disagreement": rule_positive and model.get("label") == "ham", "score_kind": "no_combined_probability"}
