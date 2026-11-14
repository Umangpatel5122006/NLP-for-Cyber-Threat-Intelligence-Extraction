# Evaluation Report: crf_tagger (Layer: entities)

Metrics version: 2

## 1. Headline Metrics (All-Sentence End-to-End)
- **Total Sentences Evaluated**: 1238

| Metric Mode | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Exact-Span (seqeval default) | 0.247 | 0.274 | **0.260** |
| Exact-Span (strict IOB2) | 0.247 | 0.274 | 0.260 |
| Relaxed-Span (Overlap) | 0.310 | 0.343 | 0.326 |
| Token-Level (Macro) | 0.240 | 0.364 | 0.286 |

### Per-Label Exact-Span Performance
| Entity Label | Precision | Recall | F1 Score | Gold Support | Predicted Spans |
|---|---:|---:|---:|---:|---:|
| `INFRASTRUCTURE` | 0.208 | 0.229 | 0.218 | 406 | 447 |
| `MALWARE_TOOL` | 0.223 | 0.218 | 0.220 | 303 | 296 |
| `THREAT_ACTOR` | 0.342 | 0.454 | 0.390 | 194 | 257 |

## 2. Diagnostic Metrics (Relevant Sentences Only)
- **Relevant Sentences**: 434 / 1238
> [!NOTE]
> Relevant-only metrics evaluate model precision/recall solely on ground-truth relevant sentences.
> They isolate tagger capabilities from sentence relevance filtering errors.

| Metric Mode | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Exact-Span (seqeval default) | 0.440 | 0.274 | **0.337** |
| Relaxed-Span (Overlap) | 0.553 | 0.343 | 0.423 |
| Token-Level (Macro) | 0.471 | 0.364 | 0.410 |

### Per-Label Diagnostic (Relevant-Only)
| Entity Label | Precision | Recall | F1 Score | Gold Support | Predicted Spans |
|---|---:|---:|---:|---:|---:|
| `INFRASTRUCTURE` | 0.397 | 0.229 | 0.291 | 406 | 234 |
| `MALWARE_TOOL` | 0.371 | 0.218 | 0.274 | 303 | 178 |
| `THREAT_ACTOR` | 0.591 | 0.454 | 0.513 | 194 | 149 |

## 3. Error Taxonomy Breakdown (All Sentences)
| Error Category | Count | Percentage of Gold/Pred | Description |
|---|---:|---:|---|
| Exact Match | 247 | - | Identical label and token boundaries |
| Boundary Error | 63 | - | Matching label & overlapping span, but boundary mismatch |
| Wrong Type (Confusion) | 123 | - | Overlapping span, but conflicting entity labels |
| Missed (False Negative) | 470 | - | Gold entity completely missed by model |
| Spurious (False Positive) | 567 | - | Predicted entity with no overlapping gold span |

## 4. Per-Source Breakdown
| Source | Sentences | Exact P | Exact R | Exact F1 | Relaxed F1 | Token Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| `aptnotes` | 155 | 0.266 | 0.318 | 0.290 | 0.372 | 0.306 |
| `crowdstrike` | 437 | 0.183 | 0.342 | 0.238 | 0.296 | 0.241 |
| `mandiant` | 226 | 0.186 | 0.488 | 0.269 | 0.314 | 0.289 |
| `mitre` | 280 | 0.362 | 0.239 | 0.288 | 0.359 | 0.388 |
| `sentinel` | 140 | 0.188 | 0.138 | 0.159 | 0.249 | 0.252 |