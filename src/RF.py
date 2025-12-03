import pandas as pd
import re
import glob
import os

verbose = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _normalize(text: str) -> str:
    """Normalize odorant names for comparison."""
    if pd.isna(text):
        return ""
    t = str(text).lower()
    # Remove spaces, punctuation, hyphens, underscores, em-dashes, dots, commas
    return re.sub(r"[-_—., ]", "", t)


def find_RF(odorant: str, RF_data: pd.DataFrame) -> float | None:
    """Return the RF value matching the odorant (by Name or AKA) or None if it is not found."""
    q = _normalize(odorant)

    for _, row in RF_data.iterrows():
        if q == _normalize(row.get("Name")) or q == _normalize(row.get("AKA")):
            return row.get("RF")
    return None


def load_RF_df() -> pd.DataFrame:
    path = os.path.join(BASE_DIR, "..", "data", "RF.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"RF file not found: {path}")
    return pd.read_csv(path)


def load_pulse_df() -> pd.DataFrame:
    saved_dir = os.path.join(BASE_DIR, "..", "saved")
    csv_files = glob.glob(os.path.join(saved_dir, "*.csv"))

    if verbose:
        print(f"CSV files loaded: {csv_files}")

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {saved_dir}")

    return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


def process_RF():
    pulse_df = load_pulse_df()
    RF_df = load_RF_df()

    results = []
    for _, row in pulse_df.iterrows():
        odorant = row.get("odorant", "")
        response_factor = find_RF(odorant, RF_df)

        results.append({
            "odorant": odorant,
            "RF": response_factor
        })

        if verbose:
            if response_factor is not None:
                print(f"Match found {odorant}: {response_factor}")
            else:
                print(f"No match found for {odorant}")

    return pd.DataFrame(results)



if __name__ == "__main__":    
    process_RF()
