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


input_path = Path("data/raw_abstracts_v3final.json")
chromadb_path = Path("data/chroma_db")
collection_name = "mental_health_corpus"

embedding_model = "gemini-embedding-001"
batch_delay_seconds = 1.0 #pauses between embeddings due to rate limits
upsert_batch_size = 50 # write to chroma in batches instead of one record at a time.


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
        model = embedding_model,
        contents = text,
        config = types.EmbedContentConfig(task_type='RETRIEVAL_DOCUMENT'),

    )
    return result.embeddings[0].values


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set."
        ) 

    client = genai.Client(api_key = api_key)

    print(f"Loading corpus from {input_path}...")
    records = load_corpus(input_path)
    print(f"Loaded {len(records)} records.")

    chromadb_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chromadb_path))
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Check for records already embedded, so rerunning the script
    # doesnt re-embed everything again.
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
    print(f"Collection already contains {len(existing_ids)} embedded records.")

    batch_ids, batch_embeddings, batch_documents, batch_metadata = [], [], [], []

    for i, record in enumerate(records):
        pubmed_id = record.get("pubmed_id")
        if not pubmed_id or pubmed_id in existing_ids:
            continue # skip records with no id or that are already embedded.

        document_text = build_document_text(record)
        if not document_text:
            continue # skip empty documents

        try:
            embedding = get_embedding(client, document_text)
        except Exception as e:
            print(f"Skipping {pubmed_id} due to embedding error: {e}")
            continue

        batch_ids.append(pubmed_id)
        batch_embeddings.append(embedding)
        batch_documents.append(document_text)
        batch_metadata.append({
            "pubmed_id": pubmed_id,
            "title": record.get("title") or "",
            "journal": record.get("journal") or "",
            "year": record.get("year") or "",
            "search_condition": record.get("search_condition") or "",
            "search_treatment": record.get("search_treatment") or ""
        })

        time.sleep(batch_delay_seconds)

        # Write Chroma in batches instead of one by one and print the progress
        if len(batch_ids) >= upsert_batch_size:
            collection.add(
                ids=batch_ids,
                embeddings = batch_embeddings,
                documents = batch_documents,
                metadatas = batch_metadata
            )
            print(f"Embedded and stored {i + 1}/{len(records)} records."
                f"(Collection total: {collection.count()})")
            batch_ids, batch_embeddings, batch_documents, batch_metadata = [], [], [], []

    # Write any remain records that didnt fill a full batch
    if batch_ids:
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents = batch_documents,
            metadatas = batch_metadata
        )
        
    print(f"\nDone Embedding. Collection '{collection_name}' contains {collection.count()} embedded records.")


if __name__ == "__main__":
    main()