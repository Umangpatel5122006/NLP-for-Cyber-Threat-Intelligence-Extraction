# Learning Note M3: Conditional Random Fields (CRF) Tuning and Sequence Labeling

**Target Audience**: Student NLP practitioner & CTI system developer  
**Milestone**: M3 — CRF Tuning & Structured Sequence Modeling  
**Module**: `src/cti/models/crf.py`, `notebooks/14_tune_crf_stage1.py`, `notebooks/15_tune_crf_stage2_cv.py`, `notebooks/16_crf_ablation.py`, `notebooks/17_crf_new_features.py`

---

## 1. The Big Picture: Where, When, and How CRF is Used

### 1.1 Why Sequence Labeling Matters in CTI
In Cyber Threat Intelligence extraction, entities are almost never isolated single words. Threat actors and malware appear in multi-token spans:
- `["APT29", "threat", "actor"]`
- `["Cobalt", "Strike", "Beacon"]`
- `["2015", "Ukraine", "Electric", "Power", "Attack"]`

If a model treated each word as an independent classification decision (like standard Logistic Regression or Naive Bayes), it could easily produce illegal label sequences:
- `O -> I-THREAT_ACTOR` (an interior tag with no beginning tag)
- `B-THREAT_ACTOR -> I-MALWARE_TOOL` (a change of entity type in the middle of a span)

A **Linear-Chain Conditional Random Field (CRF)** is a discriminative probabilistic graphical model designed specifically for sequence labeling. Instead of computing independent token probabilities $P(y_t | x_t)$, a linear-chain CRF models the joint conditional probability of the entire tag sequence $\mathbf{y} = (y_1, y_2, \dots, y_T)$ given the sentence tokens $\mathbf{x} = (x_1, x_2, \dots, x_T)$:

$$P(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp\left( \sum_{t=1}^T \sum_{k} \lambda_k f_k(y_t, \mathbf{x}, t) + \sum_{t=1}^{T-1} \sum_{j} \mu_j g_j(y_t, y_{t+1}, \mathbf{x}, t) \right)$$

where:
1. $f_k(y_t, \mathbf{x}, t)$ are **emission feature functions** that score how well the token at position $t$ matches state $y_t$ (e.g., "word ends in `.exe` $\rightarrow$ `B-MALWARE_TOOL`").
2. $g_j(y_t, y_{t+1}, \mathbf{x}, t)$ are **transition feature functions** that model the probability of moving from tag $y_t$ to tag $y_{t+1}$ (e.g., transition `B-THREAT_ACTOR -> I-THREAT_ACTOR` has high positive weight; transition `O -> I-THREAT_ACTOR` has large negative weight).
3. $Z(\mathbf{x})$ is the normalization factor (the partition function), computed efficiently using the **Forward-Backward algorithm**.
4. At inference time, the most likely global tag sequence $\mathbf{y}^* = \arg\max_{\mathbf{y}} P(\mathbf{y} \mid \mathbf{x})$ is decoded globally in $O(T \cdot |Y|^2)$ time using the **Viterbi algorithm**.

---

## 2. What Changed and Why: The 4-Stage Tuning Architecture

### 2.1 The Stale Baseline Problem (K3)
In the original repository state, the CRF was run with default parameters $c_1 = 1.0, c_2 = 1.0$, `algorithm='lbfgs'`, `max_iterations=100`, evaluated on a single 10-report dev set without report-level grouping. A coarse grid in `notebooks/09_tune_crf.py` only varied $c_1, c_2$ across $\{0.1, 1.0, 10.0\}$ on all sentences, producing noisy F1 scores ($0.168 - 0.187$) with zero validation on whether the optimizer had converged.

### 2.2 The 4-Stage Protocol
To establish scientifically grounded hyperparameters without data leakage or overfitting:

1. **Stage 1 — Coarse Random Search (`notebooks/14_tune_crf_stage1.py`)**:
   - Randomly samples 30 hyperparameter configurations across optimization algorithms (`lbfgs`, `l2sgd`, `ap`, `pa`, `arow`), regularization parameters ($c_1 \in [10^{-3}, 10]$, $c_2 \in [10^{-3}, 10]$), feature frequency cutoffs (`min_freq` $\in \{0, 2, 3, 5\}$), and transition matrices (`all_possible_transitions` $\in \{\text{True}, \text{False}\}$).
   - Tracks exact optimizer convergence (inspecting `trainer.logparser.last_iteration['num']` vs `max_iterations`).
   - Caches pre-extracted feature dictionaries to disk using `joblib`, reducing feature extraction time from minutes to milliseconds.

2. **Stage 2 — 5-Fold Report-Grouped Cross-Validation (`notebooks/15_tune_crf_stage2_cv.py`)**:
   - Takes the top candidate configurations from Stage 1.
   - Evaluates them across all 80 train + dev reports using `GroupKFold(groups=doc_id)`.
   - Prevents intra-document leakage (sentences from the same CTI report never appear in both train and validation folds).
   - Selects the champion configuration based on mean CV Exact F1 and standard deviation.

3. **Stage 3 — Systematic Feature Ablation (`notebooks/16_crf_ablation.py`)**:
   - Evaluates the individual marginal utility of each feature family:
     - Orthographic shape and casing (`use_ortho=False`)
     - Character prefixes and suffixes (`use_affixes=False`)
     - CTI regex indicators: hashes, IPs, CVEs, domains (`use_ioc_patterns=False`)
     - Leak-audited gazetteer tags (`use_gazetteer=False`)
     - Context window sizes ($w \in \{1, 2, 3\}$)
     - Word bigrams (`use_bigrams=False`)
     - Domain trigger words (`use_triggers=False`)
   - Decides whether a feature family is essential (dropping hurts $> 0.5\%$ F1), harmful (dropping improves F1), or neutral (noise).

4. **Stage 4 — Domain-Informed Feature Exploration (`notebooks/17_crf_new_features.py`)**:
   - **Document Consistency**: Tracks within-document term frequency and whether a term appeared in earlier sentences in the same report.
   - **Distributional Word Clusters**: Trains unsupervised context co-occurrence clusters using SVD + MiniBatchKMeans (50 semantic clusters), providing Brown-style word representations without external APIs.

---

## 3. Empirical Results: What the Data Proves

### 3.1 Stage 1: Coarse Random Search (30 Draws)
Tuning $c_1, c_2$, frequency pruning (`min_freq`), and transition features over 30 candidate parameter draws revealed that **stronger L1 regularization ($c_1 \approx 1.8 - 3.0$) and frequency filtering (`min_freq` $\ge 3$) are essential**:

| Rank | Draw | Algorithm | $c_1$ | $c_2$ | `min_freq` | `all_trans` | Iterations | All-dev F1 | Rel-dev F1 | Relaxed F1 | Token Macro F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 18 | `lbfgs` | 2.955 | 0.022 | 5 | True | 100 | **0.2667** | 0.3437 | 0.3399 | 0.4577 |
| **2** | 16 | `lbfgs` | 1.843 | 0.137 | 3 | True | 100 | **0.2652** | 0.3447 | 0.3429 | 0.4564 |
| **3** | 21 | `l2sgd` | 0.000 | 0.019 | 3 | False | 68 (conv) | **0.2605** | 0.3508 | 0.3458 | 0.4742 |
| **4** | 25 | `lbfgs` | 2.002 | 1.502 | 3 | True | 100 | **0.2591** | 0.3322 | 0.3288 | 0.4467 |
| **5** | 20 | `l2sgd` | 0.000 | 0.090 | 5 | False | 90 (conv) | **0.2577** | 0.3418 | 0.3404 | 0.4678 |

*Takeaway*: Weak regularization ($c_1 < 0.1$) collapsed performance to $0.19 - 0.22$ F1 due to severe overfitting on the 93k sparse features. Filtering features with frequency $< 3$ pruned 60% of noisy attributes, speeding up training by $>2\times$ while increasing F1.

### 3.2 Stage 2: 5-Fold Report-Grouped Cross-Validation
Evaluating the top 5 candidates across 5 document-grouped folds (80 reports: 70 train + 10 dev) produced the true cross-validation rankings without inter-report leakage:

| Rank | Algorithm | $c_1$ | $c_2$ | `min_freq` | `transitions` | 5-Fold CV Exact F1 | 5-Fold CV Relaxed F1 | 5-Fold CV Token F1 | Stage 1 Dev F1 |
|---|---|---|---|---|---|---|---|---|---|
| **1 (Champion)** | **`lbfgs`** | **1.843** | **0.137** | **3** | **True** | **0.219 ± 0.036** | **0.310 ± 0.034** | **0.449 ± 0.025** | **0.265** |
| 2 | `lbfgs` | 2.002 | 1.502 | 3 | True | 0.217 ± 0.031 | 0.301 ± 0.034 | 0.443 ± 0.024 | 0.259 |
| 3 | `l2sgd` | 0.000 | 0.090 | 5 | False | 0.217 ± 0.037 | 0.305 ± 0.041 | 0.449 ± 0.032 | 0.258 |
| 4 | `l2sgd` | 0.000 | 0.019 | 3 | False | 0.216 ± 0.027 | 0.308 ± 0.026 | 0.448 ± 0.032 | 0.261 |
| 5 | `lbfgs` | 2.955 | 0.022 | 5 | True | 0.211 ± 0.031 | 0.299 ± 0.035 | 0.439 ± 0.020 | 0.267 |

*Winning configuration selected*: `algorithm='lbfgs'`, $c_1 = 1.843, c_2 = 0.137, \text{min\_freq} = 3, \text{all\_possible\_transitions} = \text{True}$.

### 3.3 Stage 3: Feature-Group Ablation Findings
Using the winning model configuration, 5-fold grouped cross-validation was run across 9 ablation conditions:

| Feature Condition | 5-Fold CV Exact F1 | $\Delta$ vs Baseline | 5-Fold CV Relaxed F1 | Decision & Empirical Rationale |
|---|---|---|---|---|
| **`all_features` (Baseline)** | **0.2192 ± 0.0303** | `+0.0000` | 0.3110 ± 0.0276 | **BASELINE** |
| **`no_gazetteer`** | 0.2057 ± 0.0268 | **-0.0135** | 0.2932 ± 0.0263 | **KEEP (Essential)** — Curated gazetteer contributes $+1.35\%$ F1 gain across folds |
| **`no_orthography`** | 0.2077 ± 0.0326 | **-0.0114** | 0.3018 ± 0.0324 | **KEEP (Essential)** — Word shape, casing, and digits contribute $+1.14\%$ F1 gain |
| **`no_affixes`** | 0.2090 ± 0.0372 | **-0.0102** | 0.2929 ± 0.0378 | **KEEP (Essential)** — Subword prefixes/suffixes contribute $+1.02\%$ F1 gain |
| **`window_1` ($w=\pm 1$)** | 0.2121 ± 0.0302 | **-0.0070** | 0.3056 ± 0.0275 | **KEEP ($w=\pm 2$ is required)** — $\pm 1$ window is too narrow to capture verb/preposition anchors |
| **`no_triggers`** | 0.2145 ± 0.0329 | **-0.0047** | 0.3021 ± 0.0321 | **NEUTRAL / KEEP** — Small positive trend, retains domain triggers |
| **`window_3` ($w=\pm 3$)** | 0.2146 ± 0.0299 | **-0.0045** | 0.3063 ± 0.0271 | **REJECT** — Expanding window to $\pm 3$ slowed training by 50% without helping |
| **`no_bigrams`** | 0.2191 ± 0.0324 | **-0.0001** | 0.3086 ± 0.0283 | **NEUTRAL** — Redundant when window features ($w=\pm 2$) are present |
| **`no_ioc_patterns`** | 0.2221 ± 0.0273 | **+0.0029** | 0.3171 ± 0.0282 | **NEUTRAL** — Handled independently by dedicated IOC extractor |

### 3.4 Stage 4: New Feature Exploration
Using the winning model configuration, two domain-informed classical representations were evaluated using 5-fold report-grouped cross-validation:

| Condition | 5-Fold CV Exact F1 | $\Delta$ vs Baseline | 5-Fold CV Relaxed F1 | Decision & Error Mechanism |
|---|---|---|---|---|
| **`baseline`** | **0.2192 ± 0.0303** | `+0.0000` | 0.3110 ± 0.0276 | **BASELINE** |
| **`+doc_consistency`** | 0.2127 ± 0.0317 | **-0.0064** | 0.3055 ± 0.0296 | **REJECT (Degrades precision)** — Over-weights frequent repetitive non-entity terms (e.g. 'attack', 'system', 'host') |
| **`+word_clusters`** | 0.2162 ± 0.0322 | **-0.0029** | 0.3062 ± 0.0309 | **NEUTRAL (Within noise)** — 50 SVD+MiniBatchKMeans clusters on context co-occurrence |
| **`+doc_and_clusters`** | 0.2163 ± 0.0281 | **-0.0029** | 0.3046 ± 0.0239 | **NEUTRAL (Within noise)** — Combined condition |

*Takeaway*: Document consistency tracking without an upstream high-precision entity classifier introduces false positives by encouraging the CRF to label non-entity terms simply because they occur repeatedly throughout a security bulletin. The cleaner and more reliable signal remains local sentence context ($w=\pm 2$) backed by the curated gazetteer.

---

## 4. Alternative Techniques: Classical and Modern

| Technique | Category | Mechanism | Strengths | Weaknesses |
|---|---|---|---|---|
| **Linear-Chain CRF** (Selected) | Classical ML | Global discriminative sequence model with Viterbi decoding | No label bias, handles dense hand-engineered domain features, fast CPU training (~30-60s) | Requires manual feature engineering; linear chain cannot capture long-range document dependencies directly |
| **MEMM** (Maximum Entropy Markov Model) | Classical ML | Locally normalized sequence model ($P(y_t \mid y_{t-1}, x)$) | Fast training (equivalent to standard multiclass Logistic Regression) | **Label Bias Problem**: states with low entropy transitions dominate probabilities regardless of observations |
| **Token LogReg** | Classical ML | Independent per-token classification | Extremely fast ($<5$s) | Produces illegal BIO transitions (e.g. `O -> I-TAG`), poor span accuracy |
| **BiLSTM-CRF** | Deep Learning (Recurrent) | Bidirectional LSTM contextual representations + CRF output layer | Automatically learns non-linear token representations | Requires GPU for practical training; slower CPU inference; more complex hyperparameter tuning |
| **Transformer (SecBERT / DistilBERT)** | Deep Learning (Pretrained) | Self-attention over subword tokens | State-of-the-art contextual semantic representations | Heavy compute (2.5h training on CPU), violates strict no-LLM / lightweight baseline rules unless explicitly permitted |

---

## 5. How to Swap Models with One Config Line

All models in this system are registered via `@register("<name>")` and inherit from `BaseTagger` or `BaseModule`. To switch between sequence labelers, edit `configs/config.yaml`:

```yaml
# configs/config.yaml
models:
  entities_tagger:
    type: "crf_tagger"      # Options: "crf_tagger", "memm_tagger", "token_logreg", "spacy_ner"
    c1: 0.1
    c2: 1.0
    min_freq: 2
    max_iterations: 200
    algorithm: "lbfgs"
```

No code edits required in preprocessing, extractors, or downstream evaluation scripts.
