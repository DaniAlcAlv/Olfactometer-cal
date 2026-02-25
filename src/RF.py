from pathlib import Path
from typing import List, Optional, Union
import unicodedata
import re
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr
from collections import Counter
import math
import argparse

verbose = True
PATH_FILE = Path("./data/RF.json")

# --- Precompiled regexes ---
CAS_REGEX = re.compile(r"^\d{2,7}-\d{2}-\d$")
SEP_REGEX = re.compile(r"[\s\-_(),]+")
SEMVER_REGEX = re.compile(r"^\d+\.\d+\.\d+$")

GREEK_MAP = {
    "alpha": "a", "beta": "b", "gamma": "g", "delta": "d",
    "epsilon": "e", "lambda": "l", "mu": "m", "pi": "p",
    "sigma": "s", "theta": "t", "zeta": "z",
}

def normalize_name(text: Optional[Union[str, float]]) -> str:
    """
    Normalize odorant names so that variants like:
    'alpha pinene', 'a-pinene', 'apinene' all collapse to the same canonical key.
    Steps:
        1) Handle None/NaN
        2) Unicode normalization (NFKC)
        3) Lowercasing
        4) Greek names -> canonical Latin shorthand (alpha -> a)
        5) Remove separators
    """
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""

    t = unicodedata.normalize("NFKC", str(text)).lower()

    for greek, latin in GREEK_MAP.items():
        # Word boundary is fine here because we normalize and then remove separators below.
        t = re.sub(rf"\b{greek}\b", latin, t)

    t = SEP_REGEX.sub("", t)
    return t


class Odorant(BaseModel):
    name: List[str] = Field(..., min_length=1)
    CAS: str = Field(...)
    RF: float = Field(..., ge=0, le=20)

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    @field_validator("name")
    @classmethod
    def validate_names(cls, v: List[str]) -> List[str]:
        # Strip entries and ensure they are all non-empty
        cleaned = [str(name).strip() for name in v if name is not None]
        if not cleaned or not all(cleaned):
            raise ValueError("name list contains an empty or whitespace-only entry")
        return cleaned

    @field_validator("CAS")
    @classmethod
    def validate_cas(cls, v: str) -> str:
        def validate_checksum(cas: str) -> bool:
            """Validates CAS checksum. Example: 7732-18-5"""
            digits = cas.replace("-", "")
            check_digit = int(digits[-1])
            body = digits[:-1][::-1]
            total = sum((i + 1) * int(num) for i, num in enumerate(body))
            return total % 10 == check_digit

        if not CAS_REGEX.match(v):
            raise ValueError(f"Invalid CAS format (got {v} expected 2to7digits-2digits-1digit)")
        try:
            if not validate_checksum(v):
                raise ValueError(f"Invalid CAS checksum for {v}")
        except Exception:
            # Avoid bare except; if int(...) failed, or other issues—treat as invalid
            raise ValueError(f"CAS checksum validation failed for {v}")
        return v


class RFData(BaseModel):
    version: str
    date: datetime
    Odorants: List[Odorant] = Field(..., min_length=1)
    _name_to_rf: dict[str, float] = PrivateAttr(default_factory=dict)

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        if not SEMVER_REGEX.match(v):
            raise ValueError("Version must follow semantic versioning (X.Y.Z)")
        return v

    @model_validator(mode="after")
    def check_uniqueness(self) -> "RFData":
        """Ensure no duplicate CAS numbers and no cross-CAS name collisions."""
        # 1) Duplicate CAS numbers (with explicit details)
        cas_numbers = [odor.CAS for odor in self.Odorants]
        dup_cas = {cas: n for cas, n in Counter(cas_numbers).items() if n > 1}
        if dup_cas:
            details = "; ".join(f"{cas} (x{n})" for cas, n in sorted(dup_cas.items()))
            raise ValueError(f"Duplicate CAS numbers detected: {details}")

        # 2) Cross-CAS name collisions after normalization
        name_to_cas: dict[str, set[str]] = {}
        for odor in self.Odorants:
            for name in odor.name:
                key = normalize_name(name)
                if not key:
                    continue
                name_to_cas.setdefault(key, set()).add(odor.CAS)

        conflicts = {k: v for k, v in name_to_cas.items() if len(v) > 1}
        if conflicts:
            details = "; ".join(
                f"{k} -> [{', '.join(sorted(v))}]"
                for k, v in sorted(conflicts.items())
            )
            raise ValueError(
                "Duplicate odorant names detected across different CAS (after normalization): "
                + details
            )

        # 3) Build fast lookup index (skip empty keys)
        index: dict[str, float] = {}
        for odor in self.Odorants:
            for name in odor.name:
                key = normalize_name(name)
                if key:
                    # If same normalized name appears multiple times within same CAS,
                    # last one wins; cross-CAS duplicates are already prevented above.
                    index[key] = odor.RF

        self._name_to_rf = index
        return self

    def get_RF_by_compound(self, odorant: str) -> Optional[float]:
        """Fast lookup using precomputed name → RF index."""
        key = normalize_name(odorant)
        return self._name_to_rf.get(key)


    def get_RF_by_cas(self, cas: str) -> Optional[float]:
        """  Lookup RF using a CAS number.  """
        for odor in self.Odorants:
            if odor.CAS == cas:
                return odor.RF
        return None


    def get_names(self, cas: str) -> List[str]:
        """
        Return the list of raw (un-normalized) names associated with a CAS.
        Returns an empty list if the CAS is not present.
        """
        for odor in self.Odorants:
            if odor.CAS == cas:
                return odor.name
        return []


    def get_all_keys(self) -> List[str]:
        """ Return all normalized lookup keys in the fast name→RF index.Useful for debugging """
        return list(self._name_to_rf.keys())


def build_rfdata_example() -> "RFData":
    """    Build a small, valid RFData object with a couple of odorants.   """
    example_odorants: List["Odorant"] = [
        Odorant(name=["Alpha pinene", "a-pinene"], CAS="80-56-8", RF=0.35),
        Odorant(name=["Eugenol"], CAS="97-53-0", RF=0.50),
    ]

    rfdata = RFData(
        version="1.0.0",
        date=datetime.now(tz=timezone.utc),
        Odorants=example_odorants,
    )
    return rfdata

def build_rfdata_from_file(path) -> Optional[RFData]:
    import json
    try:
        with open(path, "r") as f:
            rf_data = RFData(**json.load(f))
        if verbose:
            print("RF.json is valid ✅")
        return rf_data
    except Exception as e:
        print("Validation error ❌ in RFData class")
        print(e)
        return None
        

def write_rfdata_json(rfdata: "RFData", output_path: Union[str, Path]) -> Path:
    """
    Serialize RFData to JSON and write it to output_path.
    Round-trips through model_validate_json to ensure strict validity.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize with Pydantic's JSON (ensures datetime formatting, etc.)
    serialized = rfdata.model_dump_json(indent=2)

    # Validate round-trip for safety
    RFData.model_validate_json(serialized)

    output_path.write_text(serialized, encoding="utf-8")
    return output_path




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an example RF.json file.")
    parser.add_argument(
        "--output-dir",
        default="./data/RF_example.json",
        help="Output path for the generated RF.json",
    )

    args = parser.parse_args()

    # Choose one for usage example
    rf_data = build_rfdata_from_file(PATH_FILE)
    rf_data = build_rfdata_example()

    # Example lookups
    if rf_data and verbose:
        print(rf_data.get_RF_by_compound("Alpha-pinene"))  # expect RF if present
        print(rf_data.get_RF_by_compound("a-pinene"))      # same number than before
        print(rf_data.get_RF_by_cas("7785-70-8"))          # same number than before
        print(rf_data.get_RF_by_compound("Eugenol"))       # expect RF if present
        print(rf_data.get_RF_by_compound("asdf"))          # None
        print(rf_data.get_RF_by_compound(42))              # None
    
    out = write_rfdata_json(rf_data, args.output_dir)
    print(f"RF.json written to: {out.resolve()}")


