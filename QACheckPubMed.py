"""
Quality check for PubMedIngestion output

Runs checks on raw_abstracts.json:
- Total record count and duplicate check
- Missing field counts
- Per-search-term record counts (flags thin/empty search terms)
- Keyword on-topic rate (does the abstract actually mention the condition and treatment?)
- Exports a random sample for manual reading/spot-checking

"""

import pandas as pd
import json
import random
from pathlib import Path
from collections import Counter

input_path = Path("data/raw_abstracts_v3final.json")
sample_size = 20
low_result_threshold = 10 # Flag search term pairs with fewer than 10 records.
on_topic_rate_threshold = 60 # flag any search term pairs with <60% of abstracts containing both the condition and treatment terms


# Load json records into a pandas dataframe
df = pd.read_json(input_path)
print(df.head())
print(df.shape)


def duplicate_check(df: pd.DataFrame) -> None:
    """
    Ensure every PubMed id is unique (dedupe worked correctly during ingestion.)
    """
    total_records = len(df)
    unique_ids = df['pubmed_id'].nunique()
    print(f"Total records: {total_records}")
    print(f"Unique PMIDs: {unique_ids}")
    if total_records != unique_ids:
        print(f"  WARNING: {total_records - unique_ids} duplicate PMID(s) found.")
    else:
        print("  No duplicate PMIDs found.")


def missing_field_check(df: pd.DataFrame) -> None:
    """
    Identify any missing fields in the dataframe and print counts for each field.
    """
    fields = ['pubmed_id', 'title', 'journal', 'year', 'abstract']
    print("Missing field counts:")
    count_missing = df[fields].isnull().sum()
    for field, count in count_missing.items():
        flag = "<-- MISSING" if count > 0 else ""
        print(f"  {field}: {count} missing {flag}")


def search_term_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count records per search term pair and flag any with suspiciously few results.
    """
    counts = df.groupby(['search_condition', 'search_treatment']).size().reset_index(name='record_count').sort_values(['search_condition', 'search_treatment'])
    print(f"Per-search-term record counts ({len(counts)} unique pairs).")
    for _, row in counts.iterrows():
        flag = "<-- LOW" if row['record_count'] < low_result_threshold else ""
        print(f"  {row['search_condition']} + {row['search_treatment']}: {row['record_count']} records {flag}")
    return counts


def on_topic_bool(row: pd.Series) -> bool:
    """
    Row level check: does the title or abstract contain at least one word from both the condition and treatment terms?
    Note: a real match can still score False if the paper uses a synonm or a medication name. This is a rough esitmate not a precision guarantee.
        (example: "venlafaxine" instead of "SNRI")
    """
    text = f"{row['title']} {row['abstract']}".lower()
    condition_words = row['search_condition'].lower().split()
    treatment_words = row['search_treatment'].lower().split()
    has_condition = any(word in text for word in condition_words)
    has_treatment = any(word in text for word in treatment_words)
    return has_condition and has_treatment


def export_sample_by_term(df: pd.DataFrame, condition: str, treatment: str, sample_size: int, output_path: Path) -> None:
    """
    Define a function to export a random sample for a specific search term pair for manual spot checking.
    """
    subset = df[(df['search_condition'] == condition) & (df['search_treatment'] == treatment)]
    sample_size = min(sample_size, len(subset))
    sample = subset.sample(n=sample_size, random_state=None).copy()

    # Truncate long abstracts to keep export quick to skim.
    sample['abstract'] = sample['abstract'].str.slice(0, 400) + sample['abstract'].apply(
        lambda a: "..." if len(a) > 400 else ""
    )

    cols = ['pubmed_id', 'title', 'search_condition', 'search_treatment', 'abstract']
    sample[cols].to_json(output_path, orient='records', indent=2, force_ascii=False)
    print(f"Exported {sample_size} records for {condition} + {treatment} to {output_path}")


def main():
    print("=" * 60)
    duplicate_check(df)
    missing_field_check(df)
    search_term_counts(df)
    # Calculation on-topic flag per row, then average within each search term pair to get a rate. 
    df['on_topic'] = df.apply(on_topic_bool, axis=1)
    summary = df.groupby(['search_condition', 'search_treatment'])['on_topic'].mean() * 100
    print(summary)

    # Filter and print only pairs below the threshold (60). These are worth reviewing
    low_scores = summary[summary < on_topic_rate_threshold]
    print(low_scores)
    # Manual review exports for search terms flagged.
    # NOTE: "mindfulness" was renamed to "mindfulness-based intervention" after QA review.
    export_sample_by_term(df, "generalized anxiety disorder", "exercise", sample_size, Path("data/gad_exercise_sample.json"))
    export_sample_by_term(df, "major depressive disorder", "mindfulness-based intervention", sample_size, Path("data/mdd_mindfulness_sample.json"))

    # Filter and print pairs above 90% for manual review
    high_scores = summary[summary >= 90]
    print(high_scores)

    # Manually review 2 of the highest scoring pairs
    export_sample_by_term(df, "generalized anxiety disorder", "acceptance and commitment therapy", sample_size, Path("data/gad_acceptance_commitment_sample.json"))
    export_sample_by_term(df, "major depressive disorder", "light therapy", sample_size, Path("data/mdd_light_therapy.json"))


if __name__ == "__main__":
    main()

