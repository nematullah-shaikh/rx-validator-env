"""
Simplified but pharmacologically accurate drug database.
Dose ranges sourced from standard references (BNF, Micromedex).
"""

from typing import Dict, Optional, Tuple, List

# ── Drug Registry ─────────────────────────────────────────────────────────────
# Each entry:
#   class          → pharmacological class
#   adult_dose_mg  → (min, max) per single dose, oral
#   pediatric_mg_per_kg → (min, max) per single dose, per kg body weight
#   pediatric_max_single_dose_mg → absolute cap per dose for children
#   adult_only     → not approved for pediatric use
#   contraindicated_under_age → minimum age (years)
#   frequency_doses_per_day → typical doses per day (for daily calc)

DRUG_DB: Dict[str, Dict] = {
    "amoxicillin": {
        "class": "antibiotic",
        "adult_dose_mg": (250, 500),
        "pediatric_mg_per_kg": (25, 50),       # per day, divided q8h
        "pediatric_max_single_dose_mg": 500,
        "adult_only": False,
        "contraindicated_under_age": 0,
        "frequency_doses_per_day": 3,
    },
    "ibuprofen": {
        "class": "NSAID",
        "adult_dose_mg": (200, 400),
        "pediatric_mg_per_kg": (5, 10),        # per single dose
        "pediatric_max_single_dose_mg": 400,
        "adult_only": False,
        "contraindicated_under_age": 0,
        "frequency_doses_per_day": 3,
    },
    "paracetamol": {
        "class": "analgesic_antipyretic",
        "adult_dose_mg": (500, 1000),
        "pediatric_mg_per_kg": (10, 15),       # per single dose
        "pediatric_max_single_dose_mg": 1000,
        "adult_only": False,
        "contraindicated_under_age": 0,
        "frequency_doses_per_day": 4,
    },
    "acetaminophen": {   # alias
        "class": "analgesic_antipyretic",
        "adult_dose_mg": (500, 1000),
        "pediatric_mg_per_kg": (10, 15),
        "pediatric_max_single_dose_mg": 1000,
        "adult_only": False,
        "contraindicated_under_age": 0,
        "frequency_doses_per_day": 4,
    },
    "metformin": {
        "class": "antidiabetic_biguanide",
        "adult_dose_mg": (500, 1000),
        "pediatric_mg_per_kg": None,
        "pediatric_max_single_dose_mg": None,
        "adult_only": False,
        "contraindicated_under_age": 10,
        "frequency_doses_per_day": 2,
    },
    "ciprofloxacin": {
        "class": "antibiotic_fluoroquinolone",
        "adult_dose_mg": (250, 750),
        "pediatric_mg_per_kg": None,
        "pediatric_max_single_dose_mg": None,
        "adult_only": True,
        "contraindicated_under_age": 18,
        "frequency_doses_per_day": 2,
    },
    "aspirin": {
        "class": "NSAID_antiplatelet",
        "adult_dose_mg": (75, 650),
        "pediatric_mg_per_kg": None,
        "pediatric_max_single_dose_mg": None,
        "adult_only": False,
        "contraindicated_under_age": 12,    # Reye's syndrome risk
        "frequency_doses_per_day": 1,
    },
    "warfarin": {
        "class": "anticoagulant",
        "adult_dose_mg": (1, 10),
        "pediatric_mg_per_kg": (0.1, 0.2),
        "pediatric_max_single_dose_mg": 5,
        "adult_only": False,
        "contraindicated_under_age": 0,
        "frequency_doses_per_day": 1,
    },
    "atorvastatin": {
        "class": "statin_lipid_lowering",
        "adult_dose_mg": (10, 80),
        "pediatric_mg_per_kg": None,
        "pediatric_max_single_dose_mg": None,
        "adult_only": True,
        "contraindicated_under_age": 10,
        "frequency_doses_per_day": 1,
    },
    "lisinopril": {
        "class": "ACE_inhibitor_antihypertensive",
        "adult_dose_mg": (5, 40),
        "pediatric_mg_per_kg": (0.07, 0.6),
        "pediatric_max_single_dose_mg": 40,
        "adult_only": False,
        "contraindicated_under_age": 6,
        "frequency_doses_per_day": 1,
    },
    "azithromycin": {
        "class": "antibiotic_macrolide",
        "adult_dose_mg": (250, 500),
        "pediatric_mg_per_kg": (10, 10),    # fixed 10mg/kg day 1
        "pediatric_max_single_dose_mg": 500,
        "adult_only": False,
        "contraindicated_under_age": 0,
        "frequency_doses_per_day": 1,
    },
}

INVALID_DRUGS = [
    "florgaxitol", "zythromaxin", "medicore", "healex", "curapill",
    "virocine", "inflamex", "painkillor", "fixitol", "drugfake",
]

# ── Drug-Drug Interactions ────────────────────────────────────────────────────
# (drug_a, drug_b) → {severity, description}
INTERACTIONS: Dict[Tuple[str, str], Dict] = {
    ("warfarin", "aspirin"): {
        "severity": "HIGH",
        "description": "Warfarin + Aspirin: significantly increased bleeding risk (pharmacodynamic synergism).",
    },
    ("warfarin", "ibuprofen"): {
        "severity": "HIGH",
        "description": "Warfarin + Ibuprofen: increased anticoagulant effect and GI bleeding risk.",
    },
    ("warfarin", "ciprofloxacin"): {
        "severity": "MODERATE",
        "description": "Ciprofloxacin inhibits CYP1A2 → elevated warfarin levels.",
    },
    ("ciprofloxacin", "metformin"): {
        "severity": "MODERATE",
        "description": "Ciprofloxacin may enhance hypoglycaemic effect of metformin.",
    },
    ("aspirin", "ibuprofen"): {
        "severity": "MODERATE",
        "description": "Ibuprofen may block aspirin's antiplatelet effect; additive GI toxicity.",
    },
    ("atorvastatin", "azithromycin"): {
        "severity": "MODERATE",
        "description": "Azithromycin weakly inhibits CYP3A4 → modest increase in atorvastatin exposure.",
    },
    ("lisinopril", "ibuprofen"): {
        "severity": "MODERATE",
        "description": "NSAIDs reduce antihypertensive efficacy of ACE inhibitors; risk of acute kidney injury.",
    },
}


def lookup_drug(name: str) -> Optional[Dict]:
    """Returns drug data or None if invalid/unknown."""
    key = name.strip().lower()
    if key in DRUG_DB:
        return DRUG_DB[key]
    if key in INVALID_DRUGS:
        return None
    return None


def is_valid_drug(name: str) -> bool:
    return lookup_drug(name) is not None


def get_drug_class(name: str) -> Optional[str]:
    data = lookup_drug(name)
    return data["class"] if data else None


def check_adult_dose(drug_name: str, dose_mg: float) -> str:
    """Returns 'safe', 'overdose', or 'underdose' for adult dosing."""
    data = lookup_drug(drug_name)
    if data is None:
        return "unknown"
    lo, hi = data["adult_dose_mg"]
    if dose_mg < lo:
        return "underdose"
    elif dose_mg > hi:
        return "overdose"
    return "safe"


def check_pediatric_dose(drug_name: str, dose_mg: float, weight_kg: float, age_years: int) -> str:
    """Returns 'safe', 'overdose', 'underdose', or 'contraindicated'."""
    data = lookup_drug(drug_name)
    if data is None:
        return "unknown"

    if age_years < data["contraindicated_under_age"]:
        return "contraindicated"

    if data["pediatric_mg_per_kg"] is None:
        return "contraindicated"

    lo_per_kg, hi_per_kg = data["pediatric_mg_per_kg"]
    lo = lo_per_kg * weight_kg
    hi = min(hi_per_kg * weight_kg, data["pediatric_max_single_dose_mg"])

    if dose_mg < lo:
        return "underdose"
    elif dose_mg > hi:
        return "overdose"
    return "safe"


def get_interactions(drug_names: List[str]) -> List[Dict]:
    """Check all pair-wise interactions among a list of drugs."""
    found = []
    keys = [d.strip().lower() for d in drug_names]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pair = (keys[i], keys[j])
            pair_rev = (keys[j], keys[i])
            info = INTERACTIONS.get(pair) or INTERACTIONS.get(pair_rev)
            if info:
                found.append({
                    "drugs": [keys[i], keys[j]],
                    "severity": info["severity"],
                    "description": info["description"],
                })
    return found


def recommended_adult_dose(drug_name: str) -> Optional[Tuple[float, float]]:
    data = lookup_drug(drug_name)
    return data["adult_dose_mg"] if data else None


def recommended_pediatric_dose(drug_name: str, weight_kg: float) -> Optional[Tuple[float, float]]:
    data = lookup_drug(drug_name)
    if data is None or data["pediatric_mg_per_kg"] is None:
        return None
    lo_per_kg, hi_per_kg = data["pediatric_mg_per_kg"]
    lo = lo_per_kg * weight_kg
    hi = min(hi_per_kg * weight_kg, data["pediatric_max_single_dose_mg"])
    return round(lo, 1), round(hi, 1)
