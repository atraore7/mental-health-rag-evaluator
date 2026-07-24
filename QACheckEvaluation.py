"""
Evaluation Results Analysis

Compute quantitative metrics on evaluation_results.csv:
- Match rate (excluding TBD claims, which have no ground truth)
- Match rate segmented by phrasing_type (exact-terms vs semantic vs misspelling)
- Match rate segmented by category (adversarial, comparative, etc)
- Confidence distribution
- Hallucentated citation check (should be empty for a good system)
- Actual verdict distribution
"""

import pandas as pd
from pathlib import Path

results_path = Path("data/evaluation_results.csv")


def load_results(path:Path) -> pd.DataFrame:
    return pd.read_csv(path)

def get_scored_subset(df: pd.DataFrame) -> pd.DataFrame:
    """returns rows with a real expected_verdict to score against"""
    scored = df[df["expected_verdict"] != "TBD - test against corpus"].copy()
    scored["match_bool"] = scored["expected_verdict"].str.strip() == scored["actual_verdict"].str.strip()
    return scored

def match_rate_overall(scored: pd.DataFrame) -> dict:
    """
    Overall match rate between expected_verdict and actual_verdict.
    TBD claims are excluded as there's no pre-input ground truth to 
    score against. 
    """
    return{
        "metric": "overall_match_rate_pct",
        "group": "all",
        "matched": scored["match_bool"].sum(),
        "total": len(scored),
        "value_pct": round(scored["match_bool"].mean() * 100, 1)
    }

def match_rate_by_column(scored: pd.DataFrame, column:str) -> list[dict]:
    """
    Match rate by a specific column
    """
    summary = scored.groupby(column)["match_bool"].agg(["sum", "count", "mean"])
    rows = []
    for groupValue, row in summary.iterrows():
        rows.append({
        "metric": f"match_rate_by_{column}",
        "group": groupValue,
        "matched": int(row['sum']),
        "total": int(row["count"]),
        "value_pct": round(row["mean"] * 100, 1)
        })
    return rows

def confidence_distribution(df: pd.DataFrame) -> list[dict]:
    counts = df["actual_confidence"].value_counts()
    return [
        {"metric": "confidence_distribution", 
         "group": level,
         "matched": "",
         "total": count,
         "value_pct": ""
         }
         for level, count in counts.items()
    ]

def verdict_distribution(df: pd.DataFrame) -> list[dict]:
    counts = df["actual_verdict"].value_counts()
    return [
        {"metric": "verdict_distribution", 
         "group": verdict,
         "matched": "",
         "total": count,
         "value_pct": ""
         }
         for verdict, count in counts.items()
    ]

def hallucinated_citation_count(df: pd.DataFrame) -> dict:
    flagged = df[(df["hallucinated_citations"].notna()) & (df["hallucinated_citations"] != "")]
    return{
        "metric": "hallucinated_citations", 
         "group": "all",
         "matched": len(flagged),
         "total": len(df),
         "value_pct": round(len(flagged) / len(df) * 100, 1)
    }

def get_mismatches(df:pd.DataFrame) -> pd.DataFrame:
    """
    Get all claims where match = False, for manual review.
    a mismatch does not automatically mean a system error, it could
    mean was more precise than the pre-inputed expected verdict.
    """
    mismatches = df[df["match"] == "False"][
        ["id", "treatment", "claim", "expected_verdict", "actual_verdict", "actual_citations", "actual_findings", "actual_caveat", "notes"]
    ].copy()
    mismatches["investigation_notes"] = ""
    return mismatches

def main():

    df = load_results(results_path)
    scored = get_scored_subset(df)
    print(f"Loaded {len(df)} evaluated claims.")

    summary_rows = []
    summary_rows.append(match_rate_overall(scored))
    summary_rows.extend(match_rate_by_column(scored, "phrasing_type"))
    summary_rows.extend(confidence_distribution(df))
    summary_rows.extend(verdict_distribution(df))
    summary_rows.append(hallucinated_citation_count(df))

    Path("data/evaluation").mkdir(parents=True, exist_ok=True)

    summarydf = pd.DataFrame(summary_rows)
    summaryPath = Path("data/evaluation/evaluation_summary.csv")
    summarydf.to_csv(summaryPath, index=False)
    print(f"Wrote summary ({len(summarydf)} rows) to {summaryPath}")

    mismatches = get_mismatches(df)
    mismatchesPath =  Path("data/evaluation/mismatches_raw.csv")
    mismatches.to_csv(mismatchesPath, index=False)
    print(f"Wrote {len(mismatches)} mismatches to {mismatchesPath}")

if __name__ == "__main__":
    main()