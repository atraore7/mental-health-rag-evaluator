"""
Claim Evaluation Engine

Reads claims from evaluation_claims.csv, retrieves the most relevant chunks
from the embedded PubMed corpus (ChromaDB), and asks an LLM to evaluate each
claim against the retrieved evidence and produces a structured verdict.

Retrieval - claim text is embedded with task_type=RETRIEVAL_QUERY (the
counterpart to RETRIEVAL_DOCUMENT used during corpus embedding (refer 
to ChunkEmbed.py)) 

Output - results are written to evaluation_results.csv, one row per claim,
including the actual verdict/explaination and citations along with the pre-determined
expected_verdict from the claims file.

Requires: GEMINI_API_KEY environment variable to be set.
    set GEMINI_API_KEY={key here}
    python ClaimEvaluationEngine.py
"""

import csv
import json
import os
import time
from pathlib import Path
from typing import Literal
import chromadb
from google import genai
from google.genai import types
from pydantic import BaseModel

claims_path = Path("data/claims.csv")
chromadb_path = Path("data/chroma_db")
collection_name = "mental_health_corpus"
results_path = Path("data/evaluation_results.csv")

embedding_model = "gemini-embedding-001"
#generation_model = "gemini-2.5-flash" #maximum of 20 request a day
generation_model = "gemini-2.5-flash-lite" # finish processing claims with lower version that has higher limits
topK = 8 # number of retrieved chunks to give the LLM per claim
request_delay_seconds = 1.0 # for embedding calls (higher limits, ~10M tokens/min)
generation_delay_seconds = 9.0  # for generation calls (lower limits ~10 RPM quota)


system_instructions = """
    You are a research-literature evidence evaluator. You assess whether a treatment-efficacy \
    claim about a mental health condition is supported by the biomedical abstracts provided to you. \
    You are a research synthesis tool, not a source of clinical or personal medical advice.

    Rules you must follow:
    1. Base your verdict ONLY on the provided evidence. Do not use outside knowledge and do not guess.
    2. If the evidence does not adequately address the claim, respond with "Insufficient evidence" rather \
    than guessing or extrapolating.
    3. Use "Supported with caveat" ONLY if the evidence itself limits the claim's efficacy, either through scope restrictions \
    (severity level, duration, dosage, population, or combination-only effectiveness) or through evidence that meaningfully \
    qualifies the strength of the effect itself (for example, notable dropout rates, relapse, or effect sizes described as \
    modest/moderate rather than robust). State the caveat explicitly in the "caveat" field. Do not use "Supported \
    with caveat" for information that is merely related but does not limit whether the treatment works for the claim \
    as stated (see rule 7). Specifically, none of the following count as a caveat on their own: (a) the treatment being \
    second-line or positioned after other options. (Note: if the evidence shows the treatment is only effective when used \
    in combination with another treatment, and the claim asserts standalone efficacy, this is a genuine limitation - see rule 6) \
    ; (b) the treatment's efficacy for a different condition (for example, a comorbidity) than the one in the claim; \
    (c) a statement that the treatment's effect is similar to, weaker than, or stronger than another treatment's effect \
    (unless the claim itself makes that comparison (see rule 6)). A treatment being equally or less impressive than an \
    alternative does not mean it fails to work for the claim as stated. 
    4. If retrieved evidence concerns a specific subpopulation (example: postpartum, adolescent, older adult) rather \
    than the general population implied by the claim, note this distinction explicitly as a caveat rather than \
    treating it as generalizable.
    5. Each "finding" must include the PMIDs that directly support that specific statement in its "cited_pmids" field. \
    Do not cite a PMID in "cited_pmids" unless the finding's text is actually based on that source.
    6. Distinguish standalone-treatment claims from combination-treatment claims. A treatment shown effective only as an \
    addition to another treatment is not the same as being effective on its own. 
    7. If the evidence includes information that is clinically relevant but does not bear on whether the specific claim \
    is supported (such as comorbid conditions, the treatment's line-of-therapy positioning, effectiveness for a different \
    condition, or side-effect profiles) include this in the "clinical_notes" field rather than treating it as a caveat or \
    letting it affect the verdict. The verdict must reflect ONLY whether the evidence supports the specific treatment-efficacy \
    claim as stated. Before finalizing a caveat or verdict downgrade, check: does this specific piece of evidence directly limit \
    whether the treatment works for the claim's stated condition? If not, it belongs in "clinical_notes".
    8. Calibrate confidence based on the quality and consistency of the evidence, not just the presence of a verdict. Use "High" \
    only when evidence includes well-designed studies (adequate sample size, randomized controlled design) that consistently support \
    the verdict. Use "Medium" when evidence is mixed, limited to small or pilot studies, or shows some inconsistency across sources. \
    Use "Low" when evidence is sparse, lower-quality (uncontrolled, very small samples), or conflicting.
    9. Use "Insufficient evidence" not only when no relevant evidence exists, but also when the available evidence is too low-quality, \
    poorly controlled, or inconclusive to determine whether the treatment is actually effective (even if studies discussing the treatment \
    exist). The presence of research on a topic is not the same as that research demonstrating efficacy. Do not default to "Supported \
    with caveat" simply because evidence exists; only use it when the evidence affirmatively demonstrates some effect, with identifiable \
    limitations on that effect (see rule 3).
    10. Reserve "Contradicted" for evidence that affirmatively and confidently demonstrates that treatment does NOT work (for example, \
    a well-powered study showing no statistically significant effect with narrow confidence intervals, or a study showing the treatment \
    performs the same or worse than placebo/control, or evidence that consistently demonstrates an effect in the opposite direction from what \
    the claim asserts.). Do not use "Contradicted" for evidence that is merely inconclusive, underpowered, or shows a null \
    result with wide confidence intervals, that evidence does not confidently show the treatment fails, only that this particular study could \
    not detect an effect either way. Such evidence should be classified as "Insufficient evidence" instead, since it fails to determine efficacy \
    in either direction. 
    11. When the verdict is "Insufficient evidence", ensure the "findings" describe what research exists and specifically why it falls short of \
    establishing efficacy (for example, small sample size, lack of a control group, wide confidence intervals). If no relevant evidence was \
    retrieved at all, state this explicitly in a finding (e.g., "No evidence directly addressing this claim was found in the retrieved abstracts") \
    with an empty "cited_pmids" list, rather than fabricating or forcing a connection to unrelated evidence. Do not return a verdict of \
    "Insufficient evidence" with findings that are empty or uninformative.
    """

class Finding(BaseModel):
    finding: str 
    cited_pmids: list[str]

class Verdict(BaseModel):
    verdict: Literal["Supported", "Supported with caveat", "Contradicted", "Insufficient evidence"]
    findings: list[Finding]
    caveat: str | None # Limitation of the clamims efficacy (dosing, population, duration, etc.)
    clinical_notes: str | None #relevant context outside of the claims scope. (comorbidities, related conditions, etc.)
    citations: list[str] = []
    confidence: Literal["High", "Medium", "Low"]


def load_claims(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
    
def load_existing_results(path: Path) -> set[str]:
    """
    Return claim ids already evaluated, so re-running the script doesnt re-call the LLM 
    for claims already processed.
    """
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}
    

def embed_query(client: genai.Client, text:str) -> list[float]:
    """
    Embed a claim as a query, not a document (as done in ChunkEmbed.py
    """
    result = client.models.embed_content(
        model = embedding_model,
        contents=text,
        config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values

def retrieve_evidence(collection, query_embedding: list[float], topK: int) -> list[dict]:
    """
    Query ChromaDB and return a clean list of evidence chunks with metadata.
    """
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results=topK,
        include = ["documents", "metadatas", "distances"]
    )

    evidence = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        evidence.append({
            "pubmed_id": meta.get("pubmed_id"),
            "title": meta.get("title"),
            "year": meta.get("year"),
            "text": doc,
            "distance": dist,  
        })
    return evidence

def build_prompt(claim_text: str, evidence: list[dict]) -> str:
    """
    Build the claim and retrieved evidence into one prompt for the LLM.
    """
    evidence_groups = []
    for e in evidence:
        evidence_groups.append(
            f"[PubMed ID: {e['pubmed_id']}] ({e['year']}) {e['title']} {e['text']}"
        )
    evidence_text = "\n\n--\n\n".join(evidence_groups)

    return f"""
    Claim to evaluate:
    {claim_text}

    Retrieved evidence:
    {evidence_text}

    Evaluate the claim against the evidence above, following the system instructions.
    """
    
def evaluate_claim(client: genai.Client, claim_text: str, evidence: list[dict], system_instructions) -> Verdict:
    prompt = build_prompt(claim_text, evidence)

    response = client.models.generate_content(
        model=generation_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instructions,
            response_mime_type="application/json",
            response_schema=Verdict
        ),
    )
    results = Verdict.model_validate_json(response.text)
    results.citations = list({pmid for f in results.findings for pmid in f.cited_pmids})
    return results
    
def evaluate_claim_with_retry(client, claim_text, evidence, system_instructions, max_retries=3):
    """
    Call evaluate_claim(), retrying with progressive backoff if the request
    fails due to a rate limit (RESOURCE_EXHAUSTED) or an  unavailable response rather than a real error.
    Daily quota exhaustion is detected and NOT retried.
    """
    for attempt in range(max_retries):
        try:
            return evaluate_claim(client, claim_text, evidence, system_instructions)
        except Exception as e:
            error_text = str(e)
            if "PerDay" in error_text:
                print("Daily quota exhausted.")
                raise
            if ("RESOURCE_EXHAUSTED" in error_text or "UNAVAILABLE" in error_text) and attempt < max_retries - 1:
                waitSeconds = 30 * (attempt + 1)
                print(f"  Rate limited or server overloaded, waiting {waitSeconds}s before retry...")
                time.sleep(waitSeconds)
            else:
                raise
def normalize_pmid(pmid: str) -> str:
    return pmid.replace("PMID:", "").replace("PMID", "").strip()

def get_validated_response(client, claim_id, claim_text, evidence, retrieved_pmids, system_instructions, retries: int = 0):
    """
    Check the citations returned by the model against the retrieved evidence,
    and retries the evaluation if hallucinated citations are found.
    """
    max_retries = 2
    while True:
    
        result = evaluate_claim_with_retry(client, claim_text, evidence, system_instructions)
        
        cited_pmids_normalized = {normalize_pmid(p) for p in result.citations}
        retrieved_pmids_normalized = {normalize_pmid(p) for p in retrieved_pmids}
        hallucinated_citations = cited_pmids_normalized - retrieved_pmids_normalized

        if not hallucinated_citations:
            return result, hallucinated_citations
        
        if retries >= max_retries:
            print(f"WARNING: claim {claim_id} cited PubMed Ids not in retrieved evidence: {hallucinated_citations}.")
            return result, hallucinated_citations

        correction_instructions = (
            f"The results you return contained the following hallucenated citations: {hallucinated_citations}."
            f"You may only cite PubMed IDs found in the evidence provided."
            f"Reassess the claim using only the evidence given."
        ) 
        system_instructions = system_instructions + correction_instructions
        retries += 1

class ScopeCheckResult(BaseModel):
    in_scope: bool
    detected_condition: str | None

def check_corpus_scope(client: genai.Client, claim_text: str) -> str | None:
    """
    Ask the model whether the claim concerns GAD/MDD (this is the scope of the corpus) or a different condition.
    If the claim concerns a different condition, flags this so the user knows the result
    may reflect incidental leakage rather than delibrate coverage.
    """
    prompt = f"""
    Claim: {claim_text}

    Does this claim primarily concern Generalized Anxiety Disorder (GAD) or Major Depressive Disorder (MDD)?
    Base your answer on the specific condition the claim is about, not any condition that might be
    mentioned incidentally. 
    """
    response = client.models.generate_content(
        model=generation_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ScopeCheckResult
        ),
    )
    result = ScopeCheckResult.model_validate_json(response.text)

    if not result.in_scope:
        return f""" Warning: This claim appears to concern {result.detected_condition or 'a condition'} outside this corpus's intended scope (Generalized Anxiety Disorder and Major Depressive Disorder). Any evidence retrieved may have surfaced incidentally through overlapping search terms rather than deliberate coverage."""
        
    return None

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY variable not set."
        )
    
    client = genai.Client(api_key=api_key)

    chroma_client = chromadb.PersistentClient(path=str(chromadb_path))
    collection = chroma_client.get_or_create_collection(name=collection_name)
    print(f"Loaded ChromaDB collection '{collection_name}' with {collection.count()} records.")

    claims = load_claims(claims_path)
    #temp variable to test on a reduced set of the claims
    #claims = claims[:5]
    print(f"Testing on a reduced set of {len(claims)} claims")
    #print(f"Loaded {len(claims)} claims from {claims_path}")

    already_done = load_existing_results(results_path)
    print(f"Loaded {len(already_done)} claims already evaluated in a previous run.")

    write_header = not results_path.exists()
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        field_names = [
            "id", "condition", "treatment", "claim", "phrasing_type", "expected_verdict",
            "acceptable_verdicts", "actual_verdict", "actual_findings", "actual_caveat", 
            "actual_clinical_notes", "actual_confidence", "match", "actual_citations", 
            "hallucinated_citations", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=field_names, quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writeheader()

        for claim in claims:
            claim_id = claim["id"]
            if claim_id in already_done:
                continue # skip records that have already been evaluated.

            print(f"Evaluating claim {claim_id}: {claim['claim'][:70]}...")
            try:
                query_embedding = embed_query(client, claim['claim'])
                time.sleep(request_delay_seconds)
                evidence = retrieve_evidence(collection, query_embedding, topK)
                retrieved_pmids = {e['pubmed_id'] for e in evidence}
                # Get validated evaluation results
                result, hallucinatedCitations = get_validated_response(client, claim["id"], claim["claim"], evidence, retrieved_pmids, system_instructions)
            except Exception as e:
                print(f"Error evaluating claim {claim_id}: {e}")
                continue
                
            time.sleep(generation_delay_seconds)

            
            expected = claim["expected_verdict"]
            match = "N/A - TBD expected" if "TBD" in expected else str(expected.strip() == result.verdict.strip())

            writer.writerow({
                "id": claim_id,
                "condition": claim["condition"],
                "treatment": claim["treatment"],
                "claim": claim["claim"],
                "phrasing_type": claim["phrasing_type"],
                "expected_verdict": expected,
                "acceptable_verdicts": claim["acceptable_verdicts"],
                "actual_verdict": result.verdict,
                "actual_findings": " | ".join(f"{f.finding} [{', '.join(f.cited_pmids)}]" for f in result.findings),
                "actual_caveat": result.caveat,
                "actual_clinical_notes": result.clinical_notes,
                "actual_confidence": result.confidence,
                "match": match,
                "actual_citations": '; '.join(result.citations),
                "hallucinated_citations": "; ".join(hallucinatedCitations) if hallucinatedCitations else "",
                "notes": claim["notes"]
            })
            f.flush() # write progress incrementally

            print(f"{result.verdict} (expected: {expected})")

        print(f"\nDone. Results written to {results_path}")


if __name__ == "__main__":
    main()
                
