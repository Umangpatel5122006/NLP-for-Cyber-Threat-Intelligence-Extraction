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

## 3. Alternative Techniques: Classical and Modern

| Technique | Category | Mechanism | Strengths | Weaknesses |
|---|---|---|---|---|
| **Linear-Chain CRF** (Selected) | Classical ML | Global discriminative sequence model with Viterbi decoding | No label bias, handles dense hand-engineered domain features, fast CPU training (~30-60s) | Requires manual feature engineering; linear chain cannot capture long-range document dependencies directly |
| **MEMM** (Maximum Entropy Markov Model) | Classical ML | Locally normalized sequence model ($P(y_t \mid y_{t-1}, x)$) | Fast training (equivalent to standard multiclass Logistic Regression) | **Label Bias Problem**: states with low entropy transitions dominate probabilities regardless of observations |
| **Token LogReg** | Classical ML | Independent per-token classification | Extremely fast ($<5$s) | Produces illegal BIO transitions (e.g. `O -> I-TAG`), poor span accuracy |
| **BiLSTM-CRF** | Deep Learning (Recurrent) | Bidirectional LSTM contextual representations + CRF output layer | Automatically learns non-linear token representations | Requires GPU for practical training; slower CPU inference; more complex hyperparameter tuning |
| **Transformer (SecBERT / DistilBERT)** | Deep Learning (Pretrained) | Self-attention over subword tokens | State-of-the-art contextual semantic representations | Heavy compute (2.5h training on CPU), violates strict no-LLM / lightweight baseline rules unless explicitly permitted |

---

## 4. How to Swap Models with One Config Line

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
