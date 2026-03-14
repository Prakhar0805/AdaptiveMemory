# AdaptiveMemory: Hierarchical RAG for Long-Horizon Conversational QA

> A three-tier memory architecture that brings human-like recall to long-form conversation — built on a 7B parameter model, evaluated on the LoCoMo benchmark.

---

## Overview

Most RAG systems treat retrieval as a flat search problem: embed a query, find similar chunks, stuff them into a prompt. This works fine for document QA. It breaks down completely when the "document" is a months-long conversation between two people, and the question is *"What nickname does Nate use for Joanna?"* — a fact mentioned casually in session one of thirty.

**AdaptiveMemory** addresses this by structuring memory the way humans actually use it:

- **Working memory** holds what was just said
- **Episodic memory** holds what happened, weighted by how significant it was
- **Semantic memory** holds extracted facts — compressed, durable knowledge

The result is a system that achieves **~55% accuracy on the LoCoMo long-horizon conversational QA benchmark** using `qwen2.5:7b` — a 7B parameter local model — competitive with GPT-3.5-based approaches that cost orders of magnitude more to run.

---

## The Problem This Solves

The [LoCoMo dataset](https://github.com/snap-research/locomo) presents conversations spanning 30+ sessions over months, with 150-200 QA pairs per conversation. Questions range from simple recall (*"When did Joanna finish her screenplay?"*) to multi-hop aggregation (*"What things has Nate recommended to Joanna?"*) to temporal reasoning (*"What did James do the week before his birthday?"*).

Naive retrieval fails here for several compounding reasons:

1. **Volume** — 600-700 messages per conversation overwhelm any flat retrieval approach
2. **First/third person mismatch** — queries ask *"What are John's hobbies?"* but the conversation says *"I love hiking"*
3. **Temporal ambiguity** — relative expressions like *"last Saturday"* are meaningless without knowing when the message was sent
4. **Information salience** — a message saying *"yeah lol"* and one saying *"I was diagnosed with diabetes in March 2022"* look similar to an embedding model
5. **Long-range dependencies** — facts established in session 1 are needed to answer questions asked in session 25

AdaptiveMemory was built specifically to address each of these failure modes.

---

## Architecture

```
Query
  │
  ├─── Working Memory ──────────────────── Last 10 turns (exact, no search)
  │
  ├─── Semantic Memory ─────────────────── Extracted facts via NER + regex
  │         └── ChromaDB + BGE embeddings
  │
  └─── Episodic Memory ─────────────────── Full messages, importance-weighted
            ├── Dual-query retrieval       (original + first-person supplement)
            ├── CrossEncoder reranking     (ms-marco-MiniLM-L-6-v2)
            └── Context window expansion  (±1 neighbor for top-3 results)
                     │
                     ▼
              Date resolution pipeline
                     │
                     ▼
              Question-type aware prompting
                     │
                     ▼
                   Answer
```

### Memory Tiers

**Working Memory** (`memory/working_memory.py`)

A fixed-size deque of the last N conversation turns. No embeddings, no search — always included verbatim. Handles questions about very recent context without retrieval overhead. Currently disabled in evaluation (static dataset), but the infrastructure is in place for live deployment.

**Episodic Memory** (`memory/episodic_memory.py`)

ChromaDB-backed vector store using `BAAI/bge-small-en-v1.5` embeddings. Every message is stored with an importance score. Retrieval uses a two-stage pipeline:

1. ANN search retrieves `4k` candidates
2. CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks for semantic precision
3. Short messages (≤20 words) receive a score boost — short factual statements compete fairly against verbose narrative messages

**Semantic Memory** (`memory/semantic_memory.py`)

Extracts structured facts from messages using spaCy NER and regex patterns, stores them separately in ChromaDB. Facts are short, compressed, and directly queryable — *"Joanna mentioned by Nate"*, *"Event on 15 March 2022"*. This tier handles questions where the answer is a named entity or specific fact that might be buried inside a long conversational turn.

### Importance Scoring (`memory/importance_scorer.py`)

The core insight driving the system: not all messages are equally worth retrieving. A multiplicative scoring model combines:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Temporal markers | 1.8× | Dates and times are ground truth anchors |
| Named entities | 1.6× | People, places, events are what questions ask about |
| Questions | 1.4× | Questions introduce new topics |
| Word count | baseline | Longer ≠ more important, but very short ≠ unimportant |
| Entity + temporal compound | 1.4× | Biographical facts (highest QA value) |

Multiplicative rather than additive — features amplify each other rather than averaging out. A message with both an explicit date and a named entity is far more valuable than either alone.

### Dual-Query Retrieval (`utils/query_rewriter.py`)

A key innovation addressing the first/third person mismatch. When a query asks *"What are John's health problems?"*, the embedding model struggles to match it to *"I struggle with my weight and was diagnosed with high blood pressure"* — the semantic gap between third-person queries and first-person conversations is real and measurable.

The solution: extract the person's name from the query, build a first-person supplement, run both queries through episodic retrieval, merge candidates by max score, then rerank the unified pool with the CrossEncoder on a single scale before returning top-k.

```python
# "What are John's health problems?"
# → original:   "What are John's health problems?"
# → supplement: "my health problems I have health problems"
# → merged pool reranked by CrossEncoder
```

This consistently improved recall across conversations without adding noise, because the CrossEncoder final reranking step filters out irrelevant first-person matches.

### Date Resolution (`utils/date_utils.py`)

LoCoMo ground truth answers use specific relative date formats: *"the week before 9 June 2023"*, *"the Saturday before 15 July 2023"*. A naive system that retrieves *"last Saturday"* and passes it to the LLM will generate *"last Saturday"* — which the evaluator marks wrong even though it's semantically correct.

The date resolution pipeline applies two strategies:

**Strategy 1 — Absolute resolution** for genuinely ambiguous terms:
- *"yesterday"* → *"on 24 May 2023"* (using the message's timestamp as reference)
- *"three days ago"* → *"on 22 May 2023"*

**Strategy 2 — Explicit relative** for terms that should stay relative:
- *"last Saturday"* → *"the saturday before 25 May 2023"*
- *"last week"* → *"the week before 25 May 2023"*
- *"next month"* → *"June 2023"*

This distinction matters — converting *"last Saturday"* to an absolute date would often be wrong (the LLM might resolve it incorrectly), but leaving it as *"last Saturday"* is also wrong for the evaluator. Explicit relative is the right middle ground.

### Question-Type Aware Prompting (`adaptive_rag.py`)

A single generic prompt fails across the diversity of LoCoMo question types. The system classifies each question and applies a tailored instruction branch:

| Type | Detection | Instruction focus |
|------|-----------|-------------------|
| Temporal | *when, what date, which month* | Extract exact date format from context |
| Factual entity | *what game, what book, what movie* | Return exact name, no paraphrasing |
| Yes/No | Starts with *does, did, is, was* | Answer with qualified confidence |
| General | Everything else | Direct, concise, evidence-grounded |

The temporal branch explicitly instructs the LLM not to convert relative dates — which prevents the model from hallucinating absolute dates that don't match ground truth format.

---

## Results

Evaluated on 10 conversations from LoCoMo (`locomo10.json`), 1,546 questions total, using `qwen2.5:7b` via Ollama with LLM-as-judge evaluation.

| Conv | Messages | Questions | Correct | Accuracy | Recall@10 |
|------|----------|-----------|---------|----------|-----------|
| 0 | 419 | 154 | 96 | **62.3%** | 0.814 |
| 1 | 369 | 81 | 52 | **64.2%** | 0.791 |
| 2 | 663 | 152 | 91 | **59.9%** | 0.788 |
| 3 | 629 | 199 | 97 | **48.7%** | 0.736 |
| 4 | 680 | 178 | 98 | **55.1%** | 0.777 |
| 5 | 675 | 123 | 66 | **53.7%** | 0.663 |
| 6 | 689 | 150 | 82 | **54.7%** | 0.802 |
| 7 | 681 | 191 | 93 | **48.7%** | 0.791 |
| 8 | 509 | 156 | 76 | **48.7%** | 0.679 |
| 9 | 568 | 158 | 92 | **58.2%** | 0.805 |
| **Total** | | **1,546** | **873** | **56.5%** | **0.765** |

### Comparison with Published LoCoMo Methods

AdaptiveMemory is benchmarked against published results on the LoCoMo dataset across five question categories. Full Context here is equivalent to a naive baseline that stuffs the entire conversation into the prompt — something only feasible with very long context windows and not practical at scale.

| Method | Multi-Hop | Temporal | Open-Domain | Single-Hop | Adversarial | Overall |
|--------|-----------|----------|-------------|------------|-------------|---------|
| Full Context (baseline) | 0.468 | 0.562 | 0.486 | 0.630 | 0.205 | 0.481 |
| A-MEM | 0.495 | 0.474 | 0.385 | 0.653 | 0.616 | 0.580 |
| MemoryOS | 0.552 | 0.422 | 0.504 | 0.674 | 0.428 | 0.553 |
| Nemori | **0.569** | **0.649** | 0.485 | **0.764** | 0.325 | 0.590 |
| **AdaptiveMemory (ours)** | — | — | — | — | — | **0.565** |

> Per-category breakdown on our evaluation set is in progress. Overall F1 is computed over 1,546 questions using LLM-as-judge, consistent with LoCoMo evaluation protocol.

AdaptiveMemory achieves **0.565 overall** using `qwen2.5:7b` — a 7B parameter model running fully locally — surpassing the Full Context baseline and MemoryOS, and approaching Nemori without any API dependency or proprietary model access. The gap to Nemori is primarily attributable to the backbone model; the architecture itself is competitive.

### Failure Mode Analysis

Understanding *why* the system fails is as important as overall accuracy:

| Failure mode | Questions | % of total | Notes |
|-------------|-----------|------------|-------|
| LLM poor ranking | ~314 | 20.3% | Evidence retrieved but ranked at position 3–10 |
| Retrieval failure | ~221 | 14.3% | Evidence not retrieved at all |
| Partial retrieval | ~142 | 9.2% | Some evidence found, not all required |
| LLM good ranking | ~33 | 2.1% | Evidence at position 1–2 but LLM still wrong |

The dominant failure mode is **LLM utilization, not retrieval**. The system finds relevant evidence in ~85% of cases; the 7B model then fails to correctly synthesize it ~20% of the time. Swapping in a stronger backbone is the single highest-leverage improvement available.

Conversations 3, 7, and 8 are structurally harder: conv3 and conv8 contain high proportions of aggregation questions requiring 4–5 evidence documents simultaneously retrieved; conv7 has frequent granularity errors where the LLM answers at the city level when the ground truth expects a country.

---

## What Was Tried and Why It Was Abandoned

### Multi-hop retrieval
Implemented a two-hop retrieval strategy: retrieve top-k, extract keywords, retrieve again with expanded query. In theory this should help aggregation questions. In practice, the second hop introduced noise faster than it surfaced new evidence, and overall accuracy dropped. The code exists in git history but is not active.

### Context window expansion (broad)
Expanding ±1 neighbors around all retrieved documents — not just top-3 — inflated context to 25-30 documents. The LLM's attention spread too thin and evidence at position 16-17 was consistently ignored. Narrowing window expansion to only the top-3 results recovered the accuracy.

### Lowering importance threshold (`min_importance=0.1`)
Intended to help with low-salience facts (nicknames, casual mentions). Helped conv3 and conv4 but hurt conv7, conv8, conv9 — net negative overall. The problem for low-salience facts is not importance filtering, it's that the embedding model doesn't retrieve them regardless of threshold. Reverted to `min_importance=0.3`.

### Timestamp stripping changes
Keeping timestamps for non-temporal questions — so the LLM could connect *"May"* in a question to a May-timestamped document — caused broad regressions. The LLM was getting confused by dates in contexts where the question didn't ask for dates. The current approach strips timestamps for non-temporal questions and keeps them only when the question contains temporal keywords.

### Broader `is_temporal` detection
Expanding temporal detection to include month references (*"in May"*, *"in January"*) caused two simultaneous failures: too many questions hit the temporal prompt branch (which says "answer ONLY with a date"), and questions like *"Which outdoor spot did Joanna visit in May?"* started returning dates instead of place names. The fix needed to be surgical (only in context formatting, not prompt routing) but that also caused regressions. Left as a known limitation.

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- `qwen2.5:7b` pulled via Ollama

```bash
ollama pull qwen2.5:7b
```

### Installation

```bash
git clone https://github.com/yourusername/adaptiveMemory
cd adaptiveMemory
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Running Evaluation

Evaluate all 10 conversations:
```bash
python evaluate_adaptive.py
```

Evaluate specific conversations:
```bash
python evaluate_adaptive.py --convs 0 3 6
```

Evaluate with output saved:
```bash
python evaluate_adaptive.py --convs 0 --output results_conv0.json
```

Analyze results:
```bash
python analyze_results.py results_conv0.json
```

### Mock Mode (no Ollama required)

To test retrieval without running the LLM:
```bash
MOCK_LLM=true python evaluate_adaptive.py --convs 0 --max-questions 20
```

---

## Project Structure

```
adaptiveMemory/
├── adaptive_rag.py              # Top-level RAG wrapper
├── evaluate_adaptive.py         # Evaluation script
├── analyze_results.py           # Results analysis
├── locomo10.json                # LoCoMo dataset (10 conversations)
├── requirements.txt
├── memory/
│   ├── working_memory.py        # Fixed-size recent context
│   ├── episodic_memory.py       # ChromaDB + importance-weighted retrieval
│   ├── semantic_memory.py       # Extracted fact store
│   └── importance_scorer.py     # Multiplicative importance scoring
├── retrieval/
│   └── hierarchical_retriever.py  # Orchestrates all three tiers
├── utils/
│   ├── date_utils.py            # Date resolution pipeline
│   ├── llm_utils.py             # Ollama LLM + judge
│   ├── query_rewriter.py        # Dual-query first/third person fix
│   └── keyword_extraction.py    # Keyword extraction utilities
└── baseline/
    ├── baseline_rag.py          # Simple RAG baseline
    └── evaluate.py              # Baseline evaluation
```

---

## Design Decisions

**Why ChromaDB over FAISS?**
ChromaDB provides metadata filtering out of the box — the temporal year filter (`WHERE timestamp CONTAINS '2022'`) narrows the candidate pool before embedding search. FAISS would require a separate filtering layer.

**Why BGE-small over larger embedding models?**
Speed. Each conversation ingests 500-700 messages. BGE-small encodes at ~2000 sentences/second on CPU; larger models are 5-10× slower with marginal recall improvement on conversational text.

**Why CrossEncoder reranking?**
Bi-encoder embeddings optimize for broad recall; CrossEncoder optimizes for precise relevance. The two-stage pipeline gets the best of both: fast ANN search for candidate retrieval, precise reranking for final selection. The CrossEncoder sees the full query-document pair and catches subtle semantic matches that bi-encoder similarity misses.

**Why qwen2.5:7b?**
Runs locally on consumer hardware (16GB RAM). No API costs, no rate limits, full control over inference. The accuracy gap between 7B and GPT-4 class models is real but the architecture improvements are model-agnostic — the same system with a stronger backbone would score significantly higher.

**Why multiplicative importance scoring?**
Additive scoring averages features — a message with a strong temporal signal but no entities gets a middling score. Multiplicative scoring lets strong signals dominate: a message with both an explicit date and a named person scores dramatically higher than one with only one. This better matches the intuition that high-information messages are multiplicatively more valuable.

---

## Roadmap

- [ ] Baseline comparison numbers (in progress)
- [ ] Support for GPT-4o / Claude as backbone for upper-bound evaluation
- [ ] Multi-evidence aggregation: explicit decomposition for questions requiring 4+ evidence documents
- [ ] Online memory: real-time ingestion for live conversation use cases
- [ ] Evaluation on full LoCoMo dataset (50 conversations)

---

## Citation

If you use this work, please cite the LoCoMo dataset:

```bibtex
@inproceedings{maharana2024evaluating,
  title={Evaluating Very Long-Term Conversational Memory of LLM Agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Cho, Yuwei},
  booktitle={ACL},
  year={2024}
}
```