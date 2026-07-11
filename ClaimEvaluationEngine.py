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

claimsPath = Path("data/evaluation_claims.csv")
chromaDBPath = Path("data/chroma_db")
collectionName = "mental_health_corpus"
resultsPath = Path("evaluation_results.csv")

embeddingModel = "gemini-embedding-001"
generationModel = "gemini-2.5-flash" #maximum of 20 request a day
#generationModel = "gemini-2.5-flash-lite" # finish processing claims with lower version that has higher limits (rerun tomorrow when limit resets)
topK = 8 # number of retrieved chunks to give the LLM per claim
requestDelaySeconds = 1.0 # for embedding calls (higher limits, ~10M tokens/min)
generationDelaySeconds = 9.0  # for generation calls (lower limits ~10 RPM quota)


systemInstructions = """
    You are a research-literature evidence evaluator. You assess whether a treatment-efficacy \
    claim about a mental health condition is supported by the biomedical abstracts provided to you. \
    You are a research synthesis tool, not a source of clinical or personal medical advice.

    Rules you must follow:
    1. Base your verdict ONLY on the provided evidence. Do not use outside knowledge and do not guess.
    2. If the evidence does not adequately address the claim, respond with "Insufficient evidence" rather \
    than guessing or extrapolating.
    3. If the evidence supports the claim only partially, with a caveat, precondition, or limited scope \
    (example: only for for a specific severity level, duration of use, population, or in combination with another \
    treatment), use "Supported with caveat" and state the caveat explicitly in your explanation. Do not default \
    to a bare "Supported" that would overstate what the evidence actually shows.
    4. If retrieved evidence concerns a specific subpopulation (example: postpartum, adolescent, older adult) rather \
    than the general population implied by the claim, note this distinction explicitly rather than treating it as directly \
    generalizable.
    5. Cite the PMIDs of the specfic abstracts that support your verdict. Only cite abstracts you actually used in your reasoning.
    6. Distinguish standalone-treatment claims from combination-treatment claims. A treatment shown effective only as an \
    addition to another treatment is not the same as being effective on its own. 
    """

class Verdict(BaseModel):
    verdict: Literal["Supported", "Supported with caveat", "Contradicted", "Insufficient evidence"]
    explanation: str
    citations: list[str]
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
        model = embeddingModel,
        contents=text,
        config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values

def retrieve_evidence(collection, queryEmbedding: list[float], topK: int) -> list[dict]:
    """
    Query ChromaDB and return a clean list of evidence chunks with metadata.
    """
    results = collection.query(
        query_embeddings = [queryEmbedding],
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

def build_prompt(claimText: str, evidence: list[dict]) -> str:
    """
    Build the claim and retrieved evidence into one prompt for the LLM.
    """
    evidenceBlocks = []
    for e in evidence:
        evidenceBlocks.append(
            f"[PubMed ID: {e['pubmed_id']}] ({e['year']}) {e['title']} {e['text']}"
        )
    evidenceText = "\n\n--\n\n".join(evidenceBlocks)

    return f"""
    Claim to evaluate:
    {claimText}

    Retrieved evidence:
    {evidenceText}

    Evaluate the claim against the evidence above, following the system instructions.
    """
    
def evaluate_claim(client: genai.Client, claimText: str, evidence: list[dict], systemInstructions) -> Verdict:
    prompt = build_prompt(claimText, evidence)

    response = client.models.generate_content(
        model=generationModel,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=systemInstructions,
            response_mime_type="application/json",
            response_schema=Verdict
        ),
    )
    return Verdict.model_validate_json(response.text)
    
def evaluate_claim_with_retry(client, claimText, evidence, systemInstructions, maxRetries=3):
    """
    Call evaluate_claim(), retrying with progressive backoff if the request
    fails due to a rate limit (RESOURCE_EXHAUSTED) or an  unavailable response rather than a real error.
    Daily quota exhaustion is detected and NOT retried.
    """
    for attempt in range(maxRetries):
        try:
            return evaluate_claim(client, claimText, evidence, systemInstructions)
        except Exception as e:
            error_text = str(e)
            if "PerDay" in error_text:
                print("Daily quota exhausted.")
                raise
            if ("RESOURCE_EXHAUSTED" in error_text or "UNAVAILABLE" in error_text) and attempt < maxRetries - 1:
                waitSeconds = 30 * (attempt + 1)
                print(f"  Rate limited or server overloaded, waiting {waitSeconds}s before retry...")
                time.sleep(waitSeconds)
            else:
                raise
def normalize_pmid(pmid: str) -> str:
    return pmid.replace("PMID:", "").replace("PMID", "").strip()

def get_validated_response(client, claimID, claimText, evidence, retrievedPMIDs, system_instructions, retries: int = 0):
    """
    Check the citations returned by the model against the retrieved evidence,
    and retries the evaluation if hallucinated citations are found.
    """
    maxRetries = 2
    while True:
    
        result = evaluate_claim_with_retry(client, claimText, evidence, system_instructions)
        
        cited_pmids_normalized = {normalize_pmid(p) for p in result.citations}
        retrieved_pmids_normalized = {normalize_pmid(p) for p in retrievedPMIDs}
        hallucinated_citations = cited_pmids_normalized - retrieved_pmids_normalized

        if not hallucinated_citations:
            return result, hallucinated_citations
        
        if retries >= maxRetries:
            print(f"WARNING: claim {claimID} cited PubMed Ids not in retrieved evidence: {hallucinated_citations}.")
            return result, hallucinated_citations

        correction_instructions = (
            f"The results you return contained the following hallucenated citations: {hallucinated_citations}."
            f"You may only cite PubMed IDs found in the evidence provided."
            f"Reassess the claim using only the evidence given."
        ) 
        system_instructions = systemInstructions + correction_instructions
        retries += 1



def main():
    apiKey = os.environ.get("GEMINI_API_KEY")
    if not apiKey:
        raise RuntimeError(
            "GEMINI_API_KEY variable not set."
            "Run: export GEMINI_API_KEY={input key here}"
        )
    
    client = genai.Client(api_key=apiKey)

    chromaClient = chromadb.PersistentClient(path=str(chromaDBPath))
    collection = chromaClient.get_or_create_collection(name=collectionName)
    print(f"Loaded ChromaDB collection '{collectionName}' with {collection.count()} records.")

    claims = load_claims(claimsPath)
    #temp variable to test on a reduced set of the claims
    claims = claims[:5]
    print(f"Testing on a reduced set of {len(claims)} claims")
    #print(f"Loaded {len(claims)} claims from {claimsPath}")

    alreadyDone = load_existing_results(resultsPath)
    print(f"Loaded {len(alreadyDone)} claims already evaluated in a previous run.")

    writeHeader = not resultsPath.exists()
    with open(resultsPath, "a", newline="", encoding="utf-8") as f:
        fieldNames = [
            "id", "condition", "treatment", "claim", "phrasing_type", "category", "expected_verdict",
            "actual_verdict", "actual_explanation", "actual_citations", "actual_confidence", "match",
            "hallucinated_citations", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldNames, quoting=csv.QUOTE_MINIMAL)
        if writeHeader:
            writer.writeheader()

        for claim in claims:
            claimId = claim["id"]
            if claimId in alreadyDone:
                continue # skip records that have already been evaluated.

            print(f"Evaluating claim {claimId}: {claim['claim'][:70]}...")
            try:
                queryEmbedding = embed_query(client, claim['claim'])
                time.sleep(requestDelaySeconds)
                evidence = retrieve_evidence(collection, queryEmbedding, topK)
                retrievedPMIDs = {e['pubmed_id'] for e in evidence}
                # Get validated evaluation results
                result, hallucinatedCitations = get_validated_response(client, claim["id"], claim["claim"], evidence, retrievedPMIDs, systemInstructions)
            except Exception as e:
                print(f"Error evaluating claim {claimId}: {e}")
                continue
                
            time.sleep(generationDelaySeconds)

            
            expected = claim["expected_verdict"]
            match = "N/A - TBD expected" if "TBD" in expected else str(expected.strip() == result.verdict.strip())

            writer.writerow({
                "id": claimId,
                "condition": claim["condition"],
                "treatment": claim["treatment"],
                "claim": claim["claim"],
                "phrasing_type": claim["phrasing_type"],
                "category": claim["category"],
                "expected_verdict": expected,
                "actual_verdict": result.verdict,
                "actual_explanation": result.explanation,
                "actual_citations": "; ".join(result.citations),
                "actual_confidence": result.confidence,
                "match": match,
                "hallucinated_citations": "; ".join(hallucinatedCitations) if hallucinatedCitations else "",
                "notes": claim["notes"]
            })
            f.flush() # write progress incrementally

            print(f"{result.verdict} (expected: {expected})")

        print(f"\nDone. Results written to {resultsPath}")


if __name__ == "__main__":
    main()
            
