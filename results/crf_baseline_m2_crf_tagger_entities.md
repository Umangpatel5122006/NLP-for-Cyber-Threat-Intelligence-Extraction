# Evaluation Report: crf_tagger (Layer: entities)

## 1. Headline Metrics (All-Sentence End-to-End)
- **Total Sentences Evaluated**: 1238

| Metric Mode | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Exact-Span (seqeval default) | 0.236 | 0.282 | **0.257** |
| Exact-Span (strict IOB2) | 0.236 | 0.282 | 0.257 |
| Relaxed-Span (Overlap) | 0.309 | 0.381 | 0.341 |
| Token-Level (Macro) | 0.425 | 0.516 | 0.458 |

### Per-Label Exact-Span Performance
| Entity Label | Precision | Recall | F1 Score | Gold Support | Predicted Spans |
|---|---:|---:|---:|---:|---:|
| `INFRASTRUCTURE` | 0.206 | 0.254 | 0.228 | 406 | 499 |
| `MALWARE_TOOL` | 0.217 | 0.231 | 0.224 | 303 | 322 |
| `THREAT_ACTOR` | 0.315 | 0.423 | 0.361 | 194 | 260 |

## 2. Diagnostic Metrics (Relevant Sentences Only)
- **Relevant Sentences**: 434 / 1238
> [!NOTE]
> Relevant-only metrics evaluate model precision/recall solely on ground-truth relevant sentences.
> They isolate tagger capabilities from sentence relevance filtering errors.

| Metric Mode | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Exact-Span (seqeval default) | 0.421 | 0.282 | **0.338** |
| Relaxed-Span (Overlap) | 0.551 | 0.381 | 0.451 |
| Token-Level (Macro) | 0.590 | 0.521 | 0.551 |

### Per-Label Diagnostic (Relevant-Only)
| Entity Label | Precision | Recall | F1 Score | Gold Support | Predicted Spans |
|---|---:|---:|---:|---:|---:|
| `INFRASTRUCTURE` | 0.401 | 0.254 | 0.311 | 406 | 257 |
| `MALWARE_TOOL` | 0.343 | 0.231 | 0.276 | 303 | 204 |
| `THREAT_ACTOR` | 0.566 | 0.423 | 0.484 | 194 | 145 |

## 3. Error Taxonomy Breakdown (All Sentences)
| Error Category | Count | Percentage of Gold/Pred | Description |
|---|---:|---:|---|
| Exact Match | 255 | - | Identical label and token boundaries |
| Boundary Error | 78 | - | Matching label & overlapping span, but boundary mismatch |
| Wrong Type (Confusion) | 127 | - | Overlapping span, but conflicting entity labels |
| Missed (False Negative) | 443 | - | Gold entity completely missed by model |
| Spurious (False Positive) | 621 | - | Predicted entity with no overlapping gold span |

## 4. Per-Source Breakdown
| Source | Sentences | Exact P | Exact R | Exact F1 | Relaxed F1 | Token Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| `aptnotes` | 155 | 0.226 | 0.318 | 0.264 | 0.365 | 0.465 |
| `crowdstrike` | 437 | 0.173 | 0.335 | 0.228 | 0.276 | 0.412 |
| `mandiant` | 226 | 0.186 | 0.523 | 0.274 | 0.323 | 0.482 |
| `mitre` | 280 | 0.354 | 0.256 | 0.297 | 0.409 | 0.540 |
| `sentinel` | 140 | 0.147 | 0.121 | 0.133 | 0.242 | 0.421 |