# Mental Health Treatment Evidence Evaluator

An end-to-end pipeline that evaluates mental health treatment claims against biomedical literature at scale, using RAG-based evidence retrieval and an LLM to produce structured, cited verdicts.

> **Disclaimer:** This is a research literature synthesis tool, not a source of clinical or personal medical advice. Verdicts reflect what a curated corpus of PubMed abstracts says about a claim — they are not a substitute for consulting a healthcare provider.

---

## TL;DR

- **What it does:** Evaluates treatment-efficacy claims about Generalized Anxiety Disorder (GAD)/Major Depressive Disorder (MDD) against ~857 PubMed abstracts, returning a structured verdict (Supported / Supported with caveat / Contradicted / Insufficient evidence), a per-finding citation trail, and separate fields for genuine efficacy caveats vs. tangential clinical context.
- **Scope:** This system evaluates whether a specific treatment-efficacy claim is supported by the evidence — it does not recommend the best treatment for a patient, and is not a substitute for clinical judgment. See "Design Philosophy" below.
- **Stack:** PubMed API → Gemini embeddings → ChromaDB → Gemini LLM (structured output via Pydantic) → pandas-based evaluation analysis, wrapped in a FastAPI backend deployed on AWS EC2.
- **Result:** Evaluated on 38 hand-curated claims across multiple runs. Zero hallucinated citations across all evaluated claims (programmatically verified via a retry-and-correct loop). **75.7% strict match rate (28/37 scored claims)**, rising to **89.2%** when accounting for claims with genuinely defensible alternate verdicts identified through manual, source-level review (see "Evaluation Methodology"). That review also surfaced a recurring caveat-classification bug that resisted prompt-only fixes — see "Findings" below.

---

## Live Demo

The evaluation engine is deployed as a REST API at `https://mentalhealtheval.hopto.org/docs`, where you can test `/evaluate` (single claim) and `/evaluate_batch` (CSV upload) directly. The corpus is curated around Generalized Anxiety Disorder and Major Depressive Disorder treatments, though some adjacent-condition content has leaked in via shared search terms (see Findings) — claims well outside this scope should still be handled correctly (either grounded in whatever relevant content exists, or returning "Insufficient evidence"), rather than hallucinating a connection. Deployed on AWS EC2, running as a persistent `systemd` service behind nginx with a free Let's Encrypt SSL certificate. Uses a free-tier Gemini API key, so there's no billing risk from public traffic — worst case is temporary quota exhaustion.

---

## Contents
- [Design Philosophy](#design-philosophy-claim-verification-not-clinical-decision-support)
- [Output Schema](#output-schema)
- [Pipeline](#pipeline)
- [Deployment](#deployment)
- [Cost](#cost)
- [Design Decisions](#design-decisions)
- [Evaluation Methodology](#evaluation-methodology)
- [Findings](#findings)
- [Next Steps](#next-steps)

---

## Overview

Given a claim like *"Escitalopram is effective for reducing symptoms of anxiety,"* the system:
1. Embeds the claim and retrieves the most relevant chunks from a corpus of ~857 PubMed abstracts on GAD and MDD treatments
2. Passes the claim + retrieved evidence to an LLM, which returns a structured verdict: **Supported**, **Supported with caveat**, **Contradicted**, or **Insufficient evidence** — along with per-finding citations, a caveat (if a genuine efficacy limitation exists), clinical notes (if relevant but tangential context exists), and a confidence level.
3. Logs the result for comparison against a hand-curated, pre-registered set of expected outcomes.

---

## Design Philosophy: Claim Verification, Not Clinical Decision Support

This system evaluates whether a specific treatment-efficacy claim is supported by the retrieved evidence — it does not recommend the best treatment for a patient, and it is not a substitute for clinical judgment.

This distinction shapes how `verdict` and `caveat` are determined. A treatment being second-line, more commonly used as an adjunct, or less effective than an alternative doesn't mean it fails to work — it means a clinician might reasonably choose something else first. That's clinically relevant, but it doesn't change whether the underlying efficacy claim is true. This system routes that kind of context to a separate `clinical_notes` field rather than letting it downgrade the verdict, so:

- `verdict` / `caveat` answer: does the evidence support this specific claim, and are there real limitations on that efficacy (population, dosage, duration, effect strength)?
- `clinical_notes` answers: what else is relevant context, even if it doesn't bear on this specific claim?

This distinction was refined iteratively through manual, source-level review — see "Findings" for the specific cases that shaped it.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data source | PubMed (NCBI E-utilities API) |
| Embeddings | Gemini Embeddings API (gemini-embedding-001) |
| Vector store | ChromaDB |
| LLM (evaluation/generation) | Gemini API (gemini-2.5-flash-lite) |
| API layer | FastAPI |
| Deployment | AWS EC2, systemd, nginx (reverse proxy) |
| SSL / domain | Let's Encrypt (Certbot), No-IP |
| Data validation | pandas |
| Version control | GitHub |

---

## Cost

Embeddings run on the free tier throughout. Generation/evaluation uses different tiers depending on context:

- **Local batch evaluation (38 claims):** paid tier — approximately **$0.03** per complete run, including any additional API calls triggered by the citation-hallucination retry logic (see `evaluate_claim_with_retry` in ClaimEvaluationEngine.py).
- **Live deployed API** (`https://mentalhealtheval.hopto.org`): free tier — no billing risk from public traffic. Trade-off: lower rate limits mean the live demo can occasionally return a 429 ("quota exceeded") under moderate use, since free-tier request limits are meaningfully lower than paid.
- **Embedding (one-time):** ~857 PubMed abstracts embedded via `gemini-embedding-001` on the free tier — no cost, re-run only when the corpus itself changes.
- **Models used:** `gemini-2.5-flash-lite` for generation/evaluation, `gemini-embedding-001` for embeddings.

At $0.03/run for local evaluation, the full evaluation suite can be re-run cheaply and often — which is what made the iterative, multi-round manual review process (see Findings below) practical in the first place. The live deployment intentionally uses the free tier instead, so public traffic carries no billing exposure.

---

## Output Schema

Each evaluated claim returns:

```python
class Finding(BaseModel):
    finding: str            # one specific statement from the evidence
    cited_pmids: list[str]  # PMIDs supporting that specific statement

class Verdict(BaseModel):
    verdict: Literal["Supported", "Supported with caveat", "Contradicted", "Insufficient evidence"]
    findings: list[Finding]       # each claim backed by its own citations
    caveat: str | None            # genuine limitation on efficacy (population, duration, effect strength)
    clinical_notes: str | None    # relevant but tangential context (comorbidities, treatment-line positioning)
    citations: list[str]          # derived programmatically from findings, not model-generated
    confidence: Literal["High", "Medium", "Low"]
```

`citations` is deliberately **not** filled in by the model directly — it's derived after parsing by flattening every `Finding`'s `cited_pmids`, guaranteeing it can never drift from what's actually cited in the reasoning (an earlier version had this exact bug: a citation referenced in the explanation text but missing from the citations list).

**The API layer (`API.py`) adds one additional field on top of this schema:**
```python
out_of_scope_warning: str | None   # flags claims outside GAD/MDD scope; evidence may reflect incidental leakage rather than deliberate coverage
```

---

## Pipeline

**Phase 1 — Corpus Ingestion** (PubMedIngestion.py)
Pulls abstracts from PubMed via esearch/efetch for 28 condition+treatment search term pairs across GAD and MDD, with deduplication by PubMed ID.

**Phase 2 — QA & Corpus Refinement** (QACheckPubMed.py)
Validates the ingested corpus: duplicate/missing-field checks, per-search-term record counts, and a keyword-based "on-topic rate" proxy measure used to flag search terms worth manually reviewing. This proxy measure only informs corpus QA at ingestion time; the system's actual relevance judgments, both at retrieval (embedding similarity) and evaluation (LLM reasoning), use semantic methods. This process surfaced and fixed two real issues (see Findings below).

**Phase 3 — Chunking + Embedding** (ChunkEmbed.py)
Embeds each abstract (title + text) using Gemini's embedding model with `task_type=RETRIEVAL_DOCUMENT`, storing vectors + metadata in a persistent ChromaDB collection (857 records).

**Phase 4 — Claim Evaluation Engine** (ClaimEvaluationEngine.py)
For each claim: embeds it with `task_type=RETRIEVAL_QUERY`, retrieves the top-8 most relevant chunks, and prompts an LLM (with a structured `Verdict` schema via Pydantic) to produce a verdict, per-finding citations, caveat, clinical_notes, and confidence level. Includes a programmatic retry-and-correct loop confirming every citation traces back to retrieved evidence (not hallucinated).

**Phase 5 — Analysis** (QACheckEvaluation.py)
Computes strict and lenient match rate (overall, by phrasing type), confidence/verdict distributions, and hallucination checks; exports a consolidated summary and mismatches file for manual review.

---

## Deployment

The evaluation engine is wrapped in a FastAPI backend (`API.py`) exposing `/evaluate` (single claim) and `/evaluate_batch` (CSV upload), both reusing the existing pipeline functions directly.

**Stack:** AWS EC2 (Ubuntu) → systemd (persistent process management) → nginx (reverse proxy) → Let's Encrypt/Certbot (SSL) → No-IP (custom domain, since certificates require a domain rather than a bare IP).

**A real deployment bug worth noting (one-time, resolved):** due to the local `chroma_db` folder being correctly excluded from git, cloning the repo onto the EC2 instance brought over all the code with none of the actual embedded corpus — and since ChromaDB's collection creation is silently additive (it creates an empty collection rather than erroring if none exists), every query returned zero evidence with no error at all. Caught by comparing `chroma.sqlite3` file size against what 857 real embeddings should produce (188 KB vs. an expected several MB), then resolved by transferring the pre-built local corpus directly via `scp`. This was specific to the initial deploy (a fresh clone with no existing data on the instance) rather than an ongoing risk — once the persistent corpus exists on the instance, this doesn't recur.

**API robustness:** since the deployed API is public and free-tier (see Cost), it will predictably hit quota limits under real traffic. Quota-exhaustion errors are caught explicitly and returned as HTTP 429 with a clear message telling the user to try again later, rather than an opaque 500 that looks like something is broken.

---

## Setup

```cmd
pip install -r requirements.txt
set GEMINI_API_KEY=actual-key-here

python3 PubMedIngestion.py       # Phase 1: build the corpus
python3 QACheckPubMed.py         # Phase 2: validate corpus
python3 ChunkEmbed.py            # Phase 3: embed into ChromaDB
python3 ClaimEvaluationEngine.py # Phase 4: run the evaluation
python3 QACheckEvaluation.py     # Phase 5: analyze results
```

---

## Design Decisions

### Corpus Coverage
Search terms were curated using a clinical literature review (NCBI treatment overview) rather than a systematic term-discovery process. Coverage is not guaranteed to be complete — a treatment absent from the corpus causes the system to return "Insufficient evidence," which may reflect a curation gap rather than a genuine lack of literature.

This approach also doesn't scale to additional conditions as-is: the current corpus covers two conditions (GAD, MDD) via manually curated condition+treatment search term pairs, and that pairing process grows combinatorially as more conditions and their full treatment landscapes are added — manual curation can't keep pace.

Rather than ingesting more broadly and relying on retrieval to compensate, a production version should scale by replicating the same precision-first process per condition. Keeping the corpus as free of irrelevant material as possible improves retrieval accuracy directly — every additional off-topic or loosely-related record is something the retrieval step has to correctly filter out at query time, so avoiding an overloaded corpus in the first place reduces that burden rather than shifting it downstream:

- **MeSH (Medical Subject Headings) — NLM's controlled, hierarchical vocabulary used to index PubMed — replaces manual literature review as the source of truth for identifying treatments.** MeSH terms are directly queryable through the same PubMed API this project already uses (via the `[MeSH Terms]` field tag in `esearch`), so this doesn't require a new data source — it's a more systematic way of using the API already in place. This scales the curation *process* itself, not the corpus size.
- **The exact-phrase-matching discipline (see "Search Precision" below) applies identically to every new condition**, keeping each new condition's ingestion as precise as GAD/MDD's rather than allowing precision to erode as coverage expands.
- **Hybrid retrieval and reranking (see Next Steps) become more valuable, not less, as more conditions are added** — a larger set of individually-curated corpora increases the chance of adjacent-condition content being retrieved (e.g., GAD and panic disorder evidence overlapping), and reranking helps ensure the most condition-specific evidence wins without needing to shrink the corpus back down.
- **A treatment with genuinely little published research should still return "Insufficient evidence"** regardless of how many conditions are covered — that reflects a real literature gap and shouldn't be engineered around.

### Search Precision / Automatic Term Mapping
Corpus ingestion uses keyword-based PubMed search, which relies on PubMed's automatic term mapping; an unquoted multi-word term (e.g., "light therapy") can silently expand into a much broader OR-query (matching "light" and "therapy" appearing anywhere, independently). This was discovered via manual sample review (a "light therapy" search initially returned mostly unrelated ketamine/ECT papers) and confirmed via PubMed's querytranslation API field. Fix: all treatment terms are wrapped in quotes to force exact-phrase matching, reducing one search term's result count from 739 to 288 while improving manually-reviewed relevance from approximately 20% to 75–80%.

### Standalone vs. Adjunct Treatment Distinction
Antipsychotics were included as a search term despite being an augmentation strategy rather than a standalone MDD treatment, specifically to test whether the system distinguishes standalone claims ("antipsychotics are an effective standalone treatment for MDD") from adjunct claims ("antipsychotics can enhance antidepressant treatment for MDD").

### Static Snapshot
The corpus is a static snapshot from ingestion time. A production system would need a scheduled refresh pipeline; the current system correctly flags "insufficient evidence" rather than guessing when coverage doesn't exist, rather than silently going stale.

### Chunking Strategy
Each abstract (title + full text) is embedded as a single chunk rather than split into sub-document chunks, since PubMed abstracts are short enough to fit well within the embedding model's context window — splitting further would add complexity without retrieval benefit for this corpus. This would need revisiting for a corpus of longer full-text articles rather than abstracts.

### Claim Set Design
The 38-claim evaluation set was designed to test more than factual accuracy:
- **Verdict format goes beyond binary Supported/Unsupported.** Several claims deliberately omit a real caveat present in the source material (e.g., benzodiazepines' short-term-use qualifier) to test whether the system surfaces it rather than returning a misleadingly bare "Supported."
- **Semantic retrieval is explicitly stress-tested.** The majority of claims deliberately use different phrasing than the ingestion search term (e.g., drug names instead of classes, brand names, acronyms in both directions, conversational phrasing, and a deliberate misspelling) with a `phrasing_type` column distinguishing these from exact-term baseline controls.
- **Standalone-vs-adjunct framing is tested directly**, using the same drug/condition pair with only the framing changed.
- **Expected verdicts are pre-registered hypotheses**, set before running the pipeline, not adjusted after seeing results — disagreements are treated as informative in either direction.

### Evaluation Engine System Instructions
The system instructions combine standard prompt-engineering practices (clear rules, explicit examples, structured output via Pydantic) with rule-level revisions driven directly by problems found through repeated testing — the current 11-rule set is the result of multiple rounds of running the evaluation, manually reviewing mismatches against source abstracts, and revising specific rules in response. Notable examples: rule 3 was tightened twice after the same tangential-caveat error (second-line positioning, cross-condition efficacy, comparative language) recurred across separate evaluation runs; rule 10 was revised after a claim's evidence showing an effect in the *opposite* direction from what was claimed was initially misclassified as "Supported with caveat" instead of "Contradicted"; rule 9 was expanded after "Insufficient evidence" was under-used when any citations existed, regardless of evidence quality. See "Findings" for the specific evidence behind each of these.

---

## Evaluation Methodology

Each claim has a pre-registered `expected_verdict`. During manual review, some claims were found to have **more than one defensible verdict** given genuinely mixed or borderline evidence (e.g., a treatment with evidence split between elderly and general-adult populations, or an "insufficient evidence" call that's arguably closer to "supported with caveat"). These are recorded in a separate `acceptable_verdicts` column, populated only after manual, source-level verification.

Two match-rate metrics are computed:
- **Strict match rate: 75.7%** — `actual_verdict == expected_verdict`
- **Lenient match rate: 89.2%** — `actual_verdict` matches either `expected_verdict` or `acceptable_verdicts`

The ~14-point gap between the two is itself informative: it separates genuine model error from cases where the evidence legitimately supports more than one reasonable conclusion.

Match rate by phrasing: 100% misspelling, 87.5% exact-term, 80.0% conversational, 69.6% semantic.

Verdict distribution across the run: 21 Supported with caveat, 8 Supported, 8 Insufficient evidence, 1 Contradicted. Confidence: 18 High, 14 Medium, 6 Low.

**A mismatch does not mean a system error** — every mismatch was manually investigated against the cited PubMed abstracts before being classified as a system finding, an expected-verdict labeling issue, or confirmation of correct behavior.

---

## Findings

### Recurring failure: tangential information misclassified as a caveat

Across three separate evaluation rounds, the model repeatedly attached a `caveat` based on content that doesn't actually limit the claim's efficacy — most often: (a) the treatment being second-line/positioned after other options, (b) efficacy for a different condition (e.g., a comorbidity), or (c) comparative language ("comparable to," "not superior to") not invited by the claim itself. Example: a Buspirone claim's caveat repeatedly cited its second-line status relative to SSRIs/SNRIs — true, but not a limitation on whether buspirone works for GAD.

**Attempted fix:** the system instructions were revised to add explicit negative examples naming these exact patterns. **Result:** the identical violation reproduced on the same claim across all three evaluation rounds following the fix. This is treated as strong evidence that the bug isn't resolvable through prompt wording alone — see "Next Steps."

### Effect-direction handling
An early version conflated "evidence shows no effect" with "evidence shows the opposite effect" — a claim asserting sleep deprivation *increases* depression symptoms, when the evidence showed it *decreases* them (an antidepressant effect), was returned as "Supported with caveat" instead of "Contradicted." Fixed by adding an explicit opposite-direction clause to the contradiction rule.

### Corpus scope leakage
Testing an out-of-scope claim ("Music cures schizophrenia") revealed the corpus contains retrievable content about schizophrenia treatment, despite schizophrenia never being a deliberate search term — likely pulled in via GAD/MDD queries that also discussed schizophrenia as a comorbid or comparison condition (e.g., shared antipsychotic treatment classes). The system correctly reasoned to "Contradicted" using this leaked content rather than hallucinating a connection. Since checking retrieved evidence's metadata tags didn't reliably catch this (a chunk tagged GAD/MDD can still discuss an unrelated condition), a dedicated scope check was added: a small model call asks whether the claim itself concerns GAD/MDD, returning an `out_of_scope_warning` field on the API response when it doesn't, so the user is alerted that results may reflect incidental leakage rather than deliberate coverage.


### Additional Positive Findings

- **"Supported with caveat" reflects genuine evidence limitations, not treatment status** — e.g., Applied Relaxation received a caveat despite being well-established and first-line, because the cited research includes real dropout/relapse data, not because it's a "lesser" treatment.
- **The system avoids redundant caveats when a claim's own wording already states its precondition** — for rTMS and Spravato, both claims already specified "treatment-resistant," and the system correctly treated this as already scoped rather than restating it as a caveat.
- **The system distinguishes genuine comparative findings from broad category assumptions** — for a sertraline-vs-venlafaxine claim, it identified a specific comparative study (older-adult cohort data) rather than defaulting to a generic "SSRIs and SNRIs are similar."
- **The system correctly identifies condition-scope mismatches between a claim and its evidence** — for a claim genericized to "anxiety" where the evidence was GAD-specific, it explicitly flagged that the evidence "primarily concerns specific anxiety disorders like GAD... rather than anxiety in general."

---

## Next Steps

- **Escalation/validation agent:** manual review across three rounds shows the tangential-caveat bug persists even after explicit prompt-level correction. The planned next step is a structural second-pass check — reviewing the generated `caveat`/`clinical_notes` split independently before returning a final result, rather than relying on generation-time instruction-following alone. This mirrors the existing citation-hallucination retry loop, which resolved a comparable problem structurally rather than through prompt wording.
- Add a faithfulness/groundedness check confirming cited abstracts actually support the specific claims made in each finding (beyond confirming citations trace to retrieved evidence, which is already verified).
- MeSH-based term identification for scaling ingestion to additional conditions, replacing manual literature review as the source of truth for identifying relevant treatments per condition (see Corpus Coverage).
- Scheduled corpus refresh pipeline (current corpus is a static snapshot).
- Elastic IP for the EC2 instance, so the domain mapping doesn't break if the instance is ever stopped/restarted.

---

## Repository Structure

```
├── API.py
├── PubMedIngestion.py
├── QACheckPubMed.py
├── ChunkEmbed.py
├── ClaimEvaluationEngine.py
├── QACheckEvaluation.py
├── requirements.txt
├── data/
│   ├── raw_abstracts.json
│   ├── chroma_db/
│   ├── claims.csv                     # 38 claims, expected_verdict + acceptable_verdicts
│   ├── evaluation_results.csv
│   └── evaluation/
│       ├── evaluation_summary.csv
│       ├── mismatches_raw.csv
│       └── manual_evaluation_of_mismatches/
│           ├── mismatches_manual_research_round1.csv
│           ├── mismatches_manual_research_round2.csv
│           └── mismatches_manual_research_round3.csv
└── README.md
```