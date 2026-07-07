# Mental Health Treatment Evidence Evaluator

An end-to-end pipeline that evaluates mental health treatment claims against biomedical literature at scale, using RAG-based evidence retrieval and an LLM to produce structured, cited verdicts.

> **Disclaimer:** This is a research literature synthesis tool, not a source of clinical or personal medical advice. Verdicts reflect what a curated corpus of PubMed abstracts says about a claim — they are not a substitute for consulting a healthcare provider.

---

## TL;DR

- **What it does:** Evaluates treatment-efficacy claims about Generalized Anxiety Disorder (GAD)/Major Depressive Disorder (MDD) against ~860 PubMed abstracts, returning a structured verdict (Supported / Supported with caveat / Contradicted / Insufficient evidence) with citations and confidence (not just a yes/no response).
- **Stack:** PubMed API → Gemini embeddings → ChromaDB → Gemini LLM (structured output via Pydantic) → pandas-based evaluation analysis.
- **Result:** Evaluated on 38 hand-curated claims. **Zero hallucinated citations** (programmatically verified). 61.3% exact match against pre-registered expectations — every mismatch manually investigated against source abstracts, several of which revealed the system was *more* precise than the original expectation.

---

## Overview

Given a claim like *"Escitalopram is effective for reducing symptoms of anxiety,"* the system:
1. Embeds the claim and retrieves the most relevant chunks from a corpus of ~860 PubMed abstracts on GAD and MDD treatments
2. Passes the claim + retrieved evidence to an LLM, which returns a structured verdict: **Supported**, **Supported with caveat**, **Contradicted**, or **Insufficient evidence** — along with an explanation and the specific PMIDs it relied on.
3. Logs the result for comparison against a hand-curated, pre-registered set of expected outcomes.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data source | PubMed (NCBI E-utilities API) |
| Embeddings | Gemini Embeddings API (gemini-embedding-001) |
| Vector store | ChromaDB |
| LLM (evaluation/generation) | Gemini API (gemini-2.5-flash / gemini-2.5-flash-lite) |
| Data validation | pandas |
| Version control | GitHub |

---

## Pipeline

**Phase 1 — Corpus Ingestion** (PubMedIngestion.py)
Pulls abstracts from PubMed via esearch/efetch for 28 condition+treatment search term pairs across GAD and MDD, with deduplication by PubMed ID.

**Phase 2 — QA & Corpus Refinement** (QACheckPubMed.py)
Validates the ingested corpus: duplicate/missing-field checks, per-search-term record counts, and a keyword-based "on-topic rate" proxy measure used to flag search terms worth manually reviewing. This proxy measure only informs corpus QA at ingestion time, the system's actual relevance judgments, both at retrieval (embedding similarity) and evaluation (LLM reasoning), use semantic methods. This process surfaced and fixed two real issues (see Design Decisions below).

**Phase 3 — Chunking + Embedding** (ChunkEmbed.py)
Embeds each abstract (title + text) using Gemini's embedding model with 'task_type=RETRIEVAL_DOCUMENT', storing vectors + metadata in a persistent ChromaDB collection (863 records).

**Phase 4 — Claim Evaluation Engine** (ClaimEvaluationEngine.py)
For each claim: embeds it with 'task_type=RETRIEVAL_QUERY', retrieves the top-8 most relevant chunks, and prompts an LLM (with a structured 'Verdict' schema via Pydantic) to produce a verdict, explanation, citations, and confidence level. Includes a programmatic check confirming every citation traces back to retrieved evidence (not hallucinated).

**Phase 5 — Analysis** (QACheckEvaluation.py)
Computes match rate (overall, by phrasing type, by category), confidence/verdict distributions, and hallucination checks; exports a consolidated summary and a mismatches file for manual review.

---

## Setup

```cmd
pip install -r requirements.txt
set GEMINI_API_KEY=actual-key-here

python3 PubMedIngestion.py       # Phase 1: build the corpus
python3 QACheckPubMed.py         # Phase 2: validate corpus
python3 ChunkEmbed.py            # Phase 3: embed into ChromaDB
python3 ClaimEvaluationEngine.py # Phase 4: run the evaluation
python3 analyze_results.py       # Phase 5: analyze results
```


---

## Design Decisions

### Corpus Coverage
Search terms were curated using a clinical literature review (NCBI treatment overview) rather than an exhaustive or automated discovery process. Coverage is not guaranteed to be complete. A treatment absent from the corpus causes the system to return "Insufficient evidence," which may reflect an ingestion gap rather than a genuine lack of literature. A production version would mitigate this via MeSH-based hierarchical term expansion or periodic review against current clinical guidelines.

### Search Precision / Automatic Term Mapping
Corpus ingestion uses keyword-based PubMed search, which relies on PubMed's automatic term mapping, an unquoted multi-word term (e.g., "light therapy") can silently expand into a much broader OR-query (matching "light" and "therapy" appearing anywhere, independently). This was discovered via manual sample review (a "light therapy" search initially returned mostly unrelated ketamine/ECT papers) and confirmed via PubMed's querytranslation API field. Fix: all treatment terms are wrapped in quotes to force exact-phrase matching, reducing one search term's result count from 739 to 288 while improving manually-reviewed relevance from approximately 20% to 75–80%.

### Standalone vs. Adjunct Treatment Distinction
Antipsychotics were included as a search term despite being an augmentation strategy rather than a standalone MDD treatment, specifically to test whether the system distinguishes standalone claims ("antipsychotics are an effective standalone treatment for MDD" — expected: condradicted) from adjunct claims ("antipsychotics can enhance antidepressant treatment for MDD" — expected: supported).

### Static Snapshot
The corpus is a static snapshot from ingestion time. A production system would need a scheduled refresh pipeline; the current system correctly flags "insufficient evidence" rather than guessing when coverage doesn't exist, rather than silently going stale.

### Claim Set Design
The 38-claim evaluation set was designed to test more than factual accuracy:
- **Verdict format goes beyond binary Supported/Unsupported.** Several claims deliberately omit a real caveat present in the source material (e.g., benzodiazepines' short-term-use qualifier) to test whether the system surfaces it rather than returning a misleadingly bare "Supported."
- **Semantic retrieval is explicitly stress-tested.** The majority of claims deliberately use different phrasing than the ingestion search term (e.g., drug names instead of classes, brand names, acronyms in both directions, conversational phrasing, and a deliberate misspelling) with a 'phrasing_type' column distinguishing these from exact-term baseline controls.
- **Standalone-vs-adjunct framing is tested directly**, using the same drug/condition pair with only the framing changed.
- **Expected verdicts are pre-registered hypotheses**, set before running the pipeline, not adjusted after seeing results, disagreements are treated as informative in either direction.

### Evaluation Engine System Instructions
Each system instruction rule maps to a specific problem found during development, not generic prompt engineering:
1. Ground verdicts only in retrieved evidence (core RAG premise)
2. Return "Insufficient evidence" rather than guessing (hallucination-resistance test)
3. Use "Supported with caveat" rather than a bare "Supported" when evidence only partially supports a claim
4. Flag subpopulation-specific evidence rather than silently generalizing it (found during corpus QA, a postpartum-depression paper surfaced under a general MDD search)
5. Cite only PMIDs actually used in reasoning
6. Distinguish standalone-treatment claims from adjunct/combination-treatment claims

---

## Evaluation Results

The system was evaluated against all 38 claims. Every mismatch between the pre-registered expected verdict and the actual output was manually investigated against the cited PubMed abstracts before being classified as a system finding, an expected-verdict labeling issue, or confirmation of correct behavior.

### Headline Results
- **Zero hallucinated citations across all 38 claims** — programmatically verified, not just spot-checked.
- **Overall match rate: 61.3% (19/31 scored claims)**, excluding 7 claims deliberately left as open questions with no pre-registered ground truth.
- Match rate by phrasing: 100% misspelling, 71.4% exact-term, 66.7% conversational, 55.0% semantic.
- Categories reaching 100% match: 'adjunct-vs-standalone', 'omitted-caveat', 'overstated', 'unrelated-treatment'.

**A mismatch does not mean a system error** — see findings below.

---

### Positive Findings (System Behaved Correctly — Original Expectations Were Often the Imprecise Part)

1. **"Supported with caveat" correctly reflects genuine evidence limitations, not treatment status.** Applied Relaxation received a caveat despite being well-established and first-line, because the cited research includes real dropout/relapse data and uncertain mechanism of action. A flat "Supported" would have overstated what the evidence actually shows. The same evidence-driven caveat behavior held for second-line treatments, confirming the system responds to what's actually in the papers rather than defaulting to hedge language based on how established a treatment is.

2. **The system correctly returns a clean "Supported" when a claim's own wording already states its precondition, rather than adding a redundant caveat.** For rTMS and Spravato, both claims already specified "treatment-resistant" the system correctly treated this as already scoped rather than restating it as a caveat. 

3. **The system correctly distinguishes genuine comparative findings from broad category assumptions.** For a sertraline-vs-venlafaxine claim, it correctly identified a specific comparative study (older-adult cohort data) rather than defaulting to a generic "SSRIs and SNRIs are similar" assumption.

4. **The system correctly identifies condition-scope mismatches between a claim and its evidence.** For a claim genericized to "anxiety," where the retrieved evidence was GAD-specific, the system explicitly noted the evidence "primarily concerns specific anxiety disorders like GAD... rather than anxiety in general."

5. **The system correctly captures genuine clinical nuance when a caveat is actually warranted.** For CBT-I, the caveat reflected legitimate methodological questions directly relevant to the claim, including that control groups receiving only antidepressants and basic sleep hygiene also showed improvement, a valid consideration for whether the effect is unique to CBT-I.

---

### Negative Findings (Real System Limitations)

1. **Explanations sometimes include irrelevant content without scoping its relevance.** In two cases (Buspirone, Bupropion), the explanation included real but off-topic findings (efficacy for a different condition, and a narrow side-effect subgroup) without clarifying these don't have bearing on the core claim.

2. **The system doesn't distinguish between caveats that qualify the core claim and caveats about a separate, secondary question.** For Escitalopram, the caveat cited findings about who might respond differently (genetic polymorphisms, oxytocin signaling, sex-specific effects, an unreplicated stress-response predictor) none of which say anything uncertain about whether escitalopram is effective. Two large network meta-analyses already answer that core question decisively.

3. **The system may under-use "Insufficient evidence" when any citations exist, regardless of quality.** For Acupuncture (deliberately excluded from search terms, incidentally surfaced via a St. John's Wort query), evidence explicitly described as "low or insufficient due to methodological limitations" still produced "Supported with caveat." This appears to happen because the retrieved evidence frames acupuncture as a known/utilized treatment, and the system seems to weight that clinical-use framing over citations describing the actual evidence of efficacy as low-quality, conflating "this is a recognized treatment option" with "this treatment has been shown to work."

---

### Future Work

**Prompt/verdict logic refinements** (identified via manual mismatch investigation):


- Distinguish "is used clinically" from "has adequate efficacy evidence" in verdict logic, to better calibrate when "Insufficient evidence" is warranted (Acupuncture finding)
- Add explicit relevance-scoping instructions to reduce off-topic content in generated explanations (Buspirone, Bupropion findings)
- Add explicit reasoning-before-verdict prompting (asking the model to reason through evidence relevance and strength before committing to a verdict), a standard prompt-engineering technique not yet incorporated.
- Add a faithfulness/groundedness verification step, a follow-up check confirming that cited abstracts actually support the specific claims made in the explanation (beyond confirming valid citations were retrieved, which is already verified programmatically)


**Corpus & retrieval scalability:**

- MeSH-based hierarchical query expansion, or broader condition-level ingestion paired with retrieval-time filtering, instead of manually encoding condition+treatment specificity into the search step, necessary for scaling beyond GAD/MDD to additional conditions.
- Scheduled corpus refresh pipeline, since the current corpus is a static snapshot from ingestion time.


**Deployment:**

- FastAPI backend wrapping the existing evaluation pipeline (/evaluate for single claims, /evaluate_batch for batch upload, preserving the project's core batch-evaluation differentiator)
- Streamlit frontend for interactive use. Re-run and re-validate the evaluation suite after the prompt refinements above, before building the deployment layer on top.

---

## Repository Structure

```
├── PubMedIngestion.py          # Phase 1: corpus ingestion
├── QACheckPubMed.py             # Phase 2: corpus QA
├── ChunkEmbed.py                # Phase 3: chunking + embedding
├── ClaimEvaluationEngine.py     # Phase 4: claim evaluation
├── analyze_results.py           # Phase 5: results analysis
├── data/
│   ├── raw_abstracts.json       # Ingested corpus
│   ├── chroma_db/                # Persistent vector store
│   ├── evaluation_claims.csv     # 38 hand-curated test claims
│   ├── evaluation_results.csv    # Full evaluation output
│   ├── evaluation_summary.csv    # Consolidated metrics
│   └── mismatches.csv            # Mismatches for manual review
└── README.md
```
