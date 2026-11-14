# Evaluation Report: crf_tagger (Layer: action)

Metrics version: 2

## 1. Headline Metrics (All-Sentence End-to-End)
- **Total Sentences Evaluated**: 1238

| Metric Mode | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Exact-Span (seqeval default) | 0.133 | 0.143 | **0.138** |
| Exact-Span (strict IOB2) | 0.133 | 0.143 | 0.138 |
| Relaxed-Span (Overlap) | 0.448 | 0.482 | 0.464 |
| Token-Level (Macro) | 0.386 | 0.406 | 0.396 |

### Per-Label Exact-Span Performance
| Entity Label | Precision | Recall | F1 Score | Gold Support | Predicted Spans |
|---|---:|---:|---:|---:|---:|
| `ACTION` | 0.133 | 0.143 | 0.138 | 608 | 654 |

## 2. Diagnostic Metrics (Relevant Sentences Only)
- **Relevant Sentences**: 434 / 1238
> [!NOTE]
> Relevant-only metrics evaluate model precision/recall solely on ground-truth relevant sentences.
> They isolate tagger capabilities from sentence relevance filtering errors.

| Metric Mode | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Exact-Span (seqeval default) | 0.238 | 0.143 | **0.179** |
| Relaxed-Span (Overlap) | 0.803 | 0.482 | 0.602 |
| Token-Level (Macro) | 0.655 | 0.406 | 0.501 |

### Per-Label Diagnostic (Relevant-Only)
| Entity Label | Precision | Recall | F1 Score | Gold Support | Predicted Spans |
|---|---:|---:|---:|---:|---:|
| `ACTION` | 0.238 | 0.143 | 0.179 | 608 | 365 |

## 3. Error Taxonomy Breakdown (All Sentences)
| Error Category | Count | Percentage of Gold/Pred | Description |
|---|---:|---:|---|
| Exact Match | 87 | - | Identical label and token boundaries |
| Boundary Error | 206 | - | Matching label & overlapping span, but boundary mismatch |
| Wrong Type (Confusion) | 0 | - | Overlapping span, but conflicting entity labels |
| Missed (False Negative) | 315 | - | Gold entity completely missed by model |
| Spurious (False Positive) | 361 | - | Predicted entity with no overlapping gold span |

## 4. Per-Source Breakdown
| Source | Sentences | Exact P | Exact R | Exact F1 | Relaxed F1 | Token Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| `aptnotes` | 155 | 0.132 | 0.170 | 0.148 | 0.548 | 0.499 |
| `crowdstrike` | 437 | 0.077 | 0.134 | 0.098 | 0.368 | 0.323 |
| `mandiant` | 226 | 0.086 | 0.136 | 0.105 | 0.479 | 0.360 |
| `mitre` | 280 | 0.221 | 0.124 | 0.159 | 0.504 | 0.439 |
| `sentinel` | 140 | 0.257 | 0.196 | 0.222 | 0.543 | 0.466 |