"""
Chunk and Embed Script
------------------------
Reads the ingested PubMed corpus(raw_abstracts_v3final.json), generates embeddings
using the Gemini API (gemini-embedding-001), and stores them in a ChromaDB collection 
for retrieval

Chunking strategy: one chunk per abtracts (title and abstract together)
Rationale: PubMed abstracts are short (under the model's 8k token input limit)
so splitting more would add complexity without benefit. 

Requires: Gemini_API_KEY variable to be set

"""

import json
import os
import time
from pathlib import Path
import chromadb
from google import genai
from google.genai import types


inputPath = Path("data/raw_abstracts_v3final.json")
chromaDBPath = Path("data/chroma_db")
collectionName = "mental_health_corpus"

embeddingModel = "gemini-embedding-001"
batchDelaySeconds = 1.0 #pauses between embeddings due to rate limits
upsertBatchSize = 50 # write to chroma in batches instead of one record at a time.


def load_corpus(path:Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def build_document_text(record:dict) -> str:
    """
    Combine title and abstract into one text block for embedding. 
    NOTE: including the title will give the embedding model useful context.
    """
    title = record.get("title") or ""
    abstract = record.get("abstract") or ""
    return f"{title}. {abstract}".strip()


def get_embedding(client: genai.Client, text: str) -> list[float]:
    """
    Generate an embedding for one document using the Gemini API.
    task_type: RETRIEVAL_DOCUMENT tells the model this text is something to be searched 
    against later, as opposed to a search query itself (Gemini's model produces slightly different vectors depending 
    on which role the text plays, and matching document-vs-query roles improves retrieval accuracy.)
    """
    result = client.models.embed_content(
        model = embeddingModel,
        contents = text,
        config = types.EmbedContentConfig(task_type='RETRIEVAL_DOCUMENT'),

    )
    return result.embeddings[0].values


def main():
    apiKey = os.environ.get("GEMINI_API_KEY")
    if not apiKey:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set."
            "Run: export GEMINI_API_KEY={input key here}"
        ) 

    client = genai.Client(api_key = apiKey)

    print(f"Loading corpus from {inputPath}...")
    records = load_corpus(inputPath)
    print(f"Loaded {len(records)} records.")

    chromaDBPath.mkdir(parents=True, exist_ok=True)
    chromaClient = chromadb.PersistentClient(path=str(chromaDBPath))
    collection = chromaClient.get_or_create_collection(name=collectionName)

    # Check for records already embedded, so rerunning the script
    # doesnt re-embed everything again.
    existingIds = set(collection.get()["ids"]) if collection.count() > 0 else set()
    print(f"Collection already contains {len(existingIds)} embedded records.")

    batchIds, batchEmbeddings, batchDocuments, batchMetadata = [], [], [], []

    for i, record in enumerate(records):
        pubmedId = record.get("pubmed_id")
        if not pubmedId or pubmedId in existingIds:
            continue # skip records with no id or that are already embedded.

        documentText = build_document_text(record)
        if not documentText:
            continue # skip empty documents

        try:
            embedding = get_embedding(client, documentText)
        except Exception as e:
            print(f"Skippig {pubmedId} due to embedding error: {e}")
            continue

        batchIds.append(pubmedId)
        batchEmbeddings.append(embedding)
        batchDocuments.append(documentText)
        batchMetadata.append({
            "pubmed_id": pubmedId,
            "title": record.get("title") or "",
            "journal": record.get("journal") or "",
            "year": record.get("year") or "",
            "search_condition": record.get("search_condition") or "",
            "search_treatment": record.get("search_treatment") or ""
        })

        time.sleep(batchDelaySeconds)

        # Write Chroma in batches instead of one by one and print the progress
        if len(batchIds) >= upsertBatchSize:
            collection.add(
                ids=batchIds,
                embeddings = batchEmbeddings,
                documents = batchDocuments,
                metadatas = batchMetadata
            )
            print(f"Embedded and stored {i + 1}/{len(records)} records."
                f"(Collection total: {collection.count()})")
            batchIds, batchEmbeddings, batchDocuments, batchMetadata = [], [], [], []

    # Write any remain records that didnt fill a full batch
    if batchIds:
        collection.add(
            ids=batchIds,
            embeddings=batchEmbeddings,
            documents = batchDocuments,
            metadatas = batchMetadata
        )
        
    print(f"\nDone Embedding. Collection '{collectionName}' contains {collection.count()} embedded records.")


if __name__ == "__main__":
    main()