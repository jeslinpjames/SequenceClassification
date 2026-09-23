"""
Fetches the UCI "Molecular Biology (Promoter Gene Sequences)" dataset
and writes it out as data/promoters.csv in the format this app expects:

    sequence,label
    tactagcaatacgcttgcgttcggtggttaagtatgtataatgcgcgggcttgtcgt,1
    ...

Usage:
    pip install ucimlrepo pandas
    python scripts/prepare_dataset.py
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo

def main():
    dataset = fetch_ucirepo(id=67)  # Molecular Biology (Promoter Gene Sequences)
    X = dataset.data.features  # 57 per-position base columns
    y = dataset.data.targets   # class column, "+" or "-"

    # Reassemble each row's 57 base columns into a single sequence string.
    sequences = X.astype(str).apply(lambda row: "".join(row.values), axis=1)
    labels = y.iloc[:, 0].map({"+": 1, "-": 0})

    out = pd.DataFrame({"sequence": sequences.str.replace(r"\s+", "", regex=True), "label": labels})
    out.to_csv("data/promoters.csv", index=False)
    print(f"Wrote {len(out)} rows to data/promoters.csv")
    print(out["label"].value_counts())

if __name__ == "__main__":
    main()
