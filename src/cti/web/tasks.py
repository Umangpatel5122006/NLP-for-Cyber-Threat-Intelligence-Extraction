"""Separate workbench task summaries; counts never imply maliciousness."""

from cti.output.merger import IOC_LABELS


TASK_SUMMARY_VERSION = 1


def summarize_tasks(document, mode, *, ioc_checked):
    if mode not in ("ner", "ioc") or type(ioc_checked) is not bool:
        raise ValueError("Task summaries require an explicit extraction mode and IOC coverage.")
    assessment = document.metadata.get("message_assessment", {})
    risk = document.metadata.get("message_risk", {})
    model = document.metadata.get("message_classifier", {})
    message_rows = [row for row in document.review_findings if row.get("kind") in ("message_rule", "message_model")]
    primary = [row for row in document.findings if row.role == "primary"]
    entities = document.metadata.get("ner", {}).get("entities", {}).get("status", "not_run")
    actions = document.metadata.get("ner", {}).get("action", {}).get("status", "not_run")
    if not ioc_checked:
        status, note = "partial_coverage", "IOC extraction was not recorded as completed. Inspect detector coverage."
    elif mode == "ioc":
        status, note = "ioc_only", "IOC patterns checked. Entity and action models were not requested in this mode."
    elif entities == actions == "ok":
        status, note = "complete", "Entities, actions and IOC patterns checked. Extracted mentions are evidence, not a threat verdict."
    elif entities == actions == "unsupported_language":
        status, note = "language_limited", "IOC patterns checked. Trained entity and action models support English only."
    else:
        status, note = "partial_coverage", "Entity or action extraction did not complete. Inspect detector coverage and available IOC evidence."
    return {"version": TASK_SUMMARY_VERSION,
            "message": {"status": assessment.get("status", "partial_coverage"),
                "summary": assessment.get("summary", "Message assessment is unavailable; missing findings do not establish safety."),
                "finding_count": len(message_rows), "rule_status": risk.get("status", "unavailable"),
                "model_status": model.get("status", "unavailable"), "model_label": model.get("label"),
                "rule_languages": ["en", "hi", "hi-en"], "trained_model_languages": ["en"],
                "language": document.metadata.get("language", "unknown")},
            "extraction": {"status": status, "mode": mode, "summary": note,
                "primary_findings": len(primary), "indicators": sum(row.label in IOC_LABELS for row in primary),
                "entities": sum(row.label not in IOC_LABELS and row.label != "ACTION" for row in primary),
                "actions": sum(row.label == "ACTION" for row in primary),
                "components": sum(row.role == "component" for row in document.findings),
                "ioc_checked": ioc_checked, "entity_model_status": entities, "action_model_status": actions,
                "trained_model_languages": ["en"], "presence_is_threat_verdict": False}}
