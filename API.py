"""
FastAPI backend wrapping the ClaimEvaluationEngine for evaluating claims.

Endpoints:
- POST /evaluate - single claim, JSON body
- POST /evaluate_batch - batch of claims, JSON list or CSV file upload

Retry/hallucination-check logic always runs (not optional) - matches the ClaimEvaluationEngine.py logic. The ClaimEvaluationEngine.py is the main entry point for the claim evaluation logic, and this API.py is a thin wrapper around it to provide a RESTful interface.
"""

import os
import csv
import io
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google import genai
import chromadb

from ClaimEvaluationEngine import (
    embed_query, retrieve_evidence, get_validated_response,
    system_instructions, chromadb_path, collection_name, topK,
    check_corpus_scope
)

app = FastAPI(title="GAD/MDD Treatment Evidence Evaluator API")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
chromadb_client = chromadb.PersistentClient(path=str(chromadb_path))
collection = chromadb_client.get_or_create_collection(name=collection_name)


#---------------------------------
# Request Response models
#--------------------------------

class EvaluateRequest(BaseModel):
    claim: str

class EvidenceItem(BaseModel):
    pubmed_id: str
    title: str
    year: str

class EvaluateResponse(BaseModel):
    claim: str
    verdict: str
    findings: list[dict]
    caveat: str | None
    clinical_notes: str | None
    citations: list[str]
    confidence: str
    hallucinated_citations: list[str]
    evidence_used: list[EvidenceItem]
    out_of_scope_warning: str | None


class BatchEvaluationResponse(BaseModel):
    results: list[EvaluateResponse]

#---------------------------------
# Evalation logic
#---------------------------------

def run_single_evaluation(claim_text: str) -> EvaluateResponse:
    """
    Run the claim evaluation engine on a single claim and return the structured response.
    """

    corpus_scope_warning = check_corpus_scope(client, claim_text)

    query_embedding = embed_query(client, claim_text)
    evidence = retrieve_evidence(collection, query_embedding, topK)
    retrieved_pubmed_ids = {e["pubmed_id"] for e in evidence}

    result, hallucinated_citations = get_validated_response(
        client, claim_id="api_call", claim_text=claim_text, evidence=evidence, 
        retrieved_pmids=retrieved_pubmed_ids, system_instructions=system_instructions
    )

    return EvaluateResponse(
        claim=claim_text,
        verdict=result.verdict,
        findings=[f.model_dump() for f in result.findings],
        caveat=result.caveat,
        clinical_notes=result.clinical_notes,
        citations=result.citations,
        confidence=result.confidence,
        hallucinated_citations=list(hallucinated_citations),
        evidence_used=[EvidenceItem(pubmed_id=e["pubmed_id"], title=e["title"], year=e["year"]) for e in evidence],
        out_of_scope_warning=corpus_scope_warning

    )

#---------------------------------
# Endpoints
#---------------------------------

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_claim(request: EvaluateRequest):
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="Claim text can't be empty.")
    return run_single_evaluation(request.claim)

@app.post("/evaluate_batch", response_model=BatchEvaluationResponse)
async def evaluate_batch(file: UploadFile = File(...)):
    """
    Evaluate a batch of claims from a CSV file. The CSV must have a 'claim' column.
    Returns a JSON list of evaluation results.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    raw = await file.read()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))

    claim_field = next((f for f in reader.fieldnames if f.strip().lower() == "claim"), None)
    if claim_field not in reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV must have a 'claim' column.")

    results = []
    for row in reader:
        claim_text = row[claim_field].strip()
        if not claim_text:
            continue  # skip empty claims
        results.append(run_single_evaluation(claim_text))
    return BatchEvaluationResponse(results=results)