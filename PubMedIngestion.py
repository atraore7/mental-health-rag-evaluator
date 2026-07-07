
"""
Pulls abstracts from PubMed for a set of condition/treatment serch terms using NCBI E-utilities API.

"""
# Import libraries
import requests
import time
import json 
import xml.etree.ElementTree as ET
from pathlib import Path


"""
 Define search terms
 Each tuple contains (condition, treatment, needs_qualifier)
 needsQualifier - True adds an extract "(treatment OR intervention OR therapy)" clause to the search query.
    used only for broad search terms like exercise that were shown in QA to have mutiple irrelevant results.
"""
SEARCH_TERMS = [
    ("generalized anxiety disorder", "cognitive behavioral therapy", False),
    ("generalized anxiety disorder", "SSRI", False),
    ("generalized anxiety disorder", "SNRI", False),
    ("generalized anxiety disorder", "benzodiazepine", False),
    ("generalized anxiety disorder", "buspirone", False),
    ("generalized anxiety disorder", "applied relaxation", False),
    ("generalized anxiety disorder", "exercise", True), # Broad term that needs a qualifier.
    ("generalized anxiety disorder", "mindfulness-based intervention", False), # updated from 'mindfulness' to 'mindfulness-based intervention' after QA review.
    ("generalized anxiety disorder", "acceptance and commitment therapy", False),
    ("major depressive disorder", "cognitive behavioral therapy", False),
    ("major depressive disorder", "interpersonal therapy", False),
    ("major depressive disorder", "SSRI", False),
    ("major depressive disorder", "SNRI", False),
    ("major depressive disorder", "electroconvulsive therapy", False),
    ("major depressive disorder", "transcranial magnetic stimulation", False),
    ("major depressive disorder", "vagus nerve stimulation", False),
    ("major depressive disorder", "esketamine", False),
    ("major depressive disorder", "exercise", True), # Broad term that needs a qualifier.
    ("major depressive disorder", "mindfulness-based intervention", False), # updated from 'mindfulness' to 'mindfulness-based intervention' after QA review.
    ("major depressive disorder", "antipsychotics", False),
    ("major depressive disorder", "tricyclic antidepressants", False),
    ("major depressive disorder", "MAOI", False),
    ("major depressive disorder", "atypical antidepressants", False),
    ("major depressive disorder", "serotonin modulators", False),
    ("major depressive disorder", "St John's Wort", False),
    ("major depressive disorder", "light therapy", False),
    ("major depressive disorder", "sleep deprivation therapy", False),
    ("major depressive disorder", "CBT-I", False)
]

# Set max PMIDs per search term to 40
results_per_query = 40 
# Set output file path and request delay
output_path = Path("data/raw_abstracts.json") # Output file path
request_delay = 0.4 #NCBI allows <=3 requests per second without an API key, this will set a delay of 0.4 seconds between requests to avoid hitting the limit

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmedIDs(condition: str, treatment: str, retmax: int, needsQualifier: bool = False) -> list[str]:
    """
    Search PubMed for a condition/treatment pair, return a list of matching PubMed IDs.
    If needsQualifier is True, restrict results to those that also mention treatment/intervention/therapy. - used to reduce topical drift for broad terms (excerise).
    """
    if needsQualifier:
        query = f'{condition} AND "{treatment}" AND (treatment OR intervention OR therapy)'
    else:  
        query = f'{condition} AND "{treatment}"'
    # Search PubMed for the search terms (condition + treatement) and return a list of pubmed ids
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json"
    }
    response = requests.get(ESEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    pubmed_ids = response.json()
    return pubmed_ids.get("esearchresult", {}).get("idlist", [])
    


def fetch_abstracts(pubmed_ids: list[str]) -> list[dict]:
    """
    Fetch the title, journal, year, and abstract for each PubMed ID.
    Returns a list of dictionaries (1 dictionary per PubMed ID)
    Papers with no abstracts are skipped.
    """
    # If there is nothing to fetch, return an empty list.
    if not pubmed_ids:
        return []
    
    params = {
        "db": "pubmed",
        "id": ",".join(pubmed_ids),
        "rettype": "abstract",
        "retmode": "xml"
    }
    response = requests.get(EFETCH_URL, params=params, timeout=30)
    response.raise_for_status()
    # Parse the XML response into a element tree for easier navigation
    root = ET.fromstring(response.content)
    
    records = []
    # One PubmedArticle element per paper.
    for article in root.findall(".//PubmedArticle"):
        pubmedID_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        journal_el = article.find(".//Journal/Title")
        year_el = article.find(".//JournalIssue/PubDate/Year")
       
       # Abstracts can have multiple sections (background, mehtods, etc)
        abstract_parts = article.findall(".//Abstract/AbstractText")
        abstract_text = " ".join(part.text or "" for part in abstract_parts if part is not None).strip()
        if not abstract_text:
            continue  # Skip articles without abstracts 

        records.append({
            "pubmed_id": pubmedID_el.text if pubmedID_el is not None else None,
            "title": title_el.text if title_el is not None else None,
            "journal": journal_el.text if journal_el is not None else None,
            "year": year_el.text if year_el is not None else None,
            "abstract": abstract_text
        })

    return records

def main():
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True) 

    # keyed by PubMed ID to deduplicate records
    all_records = {}

    for condition, treatment, needsQualifier in SEARCH_TERMS:
        print(f"Searching: {condition} + {treatment} ...")
        pubmed_ids = search_pubmedIDs(condition, treatment, results_per_query, needsQualifier)
        time.sleep(request_delay)  # Delay to avoid hitting NCBI request limit
        print(f"Found {len(pubmed_ids)} PubMed IDs for {condition} + {treatment}. Fetching abstracts...")
        records = fetch_abstracts(pubmed_ids)
        time.sleep(request_delay)  # Delay to avoid hitting NCBI request limit

        for record in records:
            record["search_condition"] = condition
            record["search_treatment"] = treatment
            record["needs_qualifier"] = needsQualifier
            all_records[record["pubmed_id"]] = record # Overwrite = audtomatic dedupe by pubmed id

        print(f"Retrieved {len(records)} abstracts with text (running total: {len(all_records)})")

    final_records = list(all_records.values()) 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(final_records)} unique abstracts to {output_path}")


if __name__ == "__main__":
    main()

# %%
