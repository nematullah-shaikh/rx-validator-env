import random
from typing import List, Tuple, Dict, Any
from models import Patient, Prescription, Action
from drug_database import (
    is_valid_drug, get_drug_class, check_adult_dose, check_pediatric_dose,
    get_interactions, recommended_adult_dose, recommended_pediatric_dose,
    DRUG_DB, INVALID_DRUGS,
)


# ─── Scenario Pools ──────────────────────────────────────────────────────────

ADULT_PATIENT = Patient(age_years=35, weight_kg=70.0, sex="male", is_pediatric=False)
ADULT_FEMALE = Patient(age_years=45, weight_kg=62.0, sex="female", is_pediatric=False)

CHILD_PATIENT_8 = Patient(age_years=8, weight_kg=25.0, sex="male", is_pediatric=True)
CHILD_PATIENT_5 = Patient(age_years=5, weight_kg=18.0, sex="female", is_pediatric=True)
CHILD_PATIENT_10 = Patient(age_years=10, weight_kg=32.0, sex="male", is_pediatric=True)


# ─── TASK 1 — Drug Identification (Easy) ─────────────────────────────────────

TASK1_SCENARIOS = [
    {
        "prescription": Prescription(drug_name="Amoxicillin", dose_mg=500, frequency="q8h", route="oral", duration_days=7),
        "answer": {"is_valid": True, "drug_class": "antibiotic"},
    },
    {
        "prescription": Prescription(drug_name="Ibuprofen", dose_mg=400, frequency="q6h", route="oral", duration_days=5),
        "answer": {"is_valid": True, "drug_class": "NSAID"},
    },
    {
        "prescription": Prescription(drug_name="Paracetamol", dose_mg=500, frequency="q6h", route="oral", duration_days=3),
        "answer": {"is_valid": True, "drug_class": "analgesic_antipyretic"},
    },
    {
        "prescription": Prescription(drug_name="Florgaxitol", dose_mg=250, frequency="q12h", route="oral", duration_days=5),
        "answer": {"is_valid": False, "drug_class": None},
    },
    {
        "prescription": Prescription(drug_name="Zythromaxin", dose_mg=500, frequency="once_daily", route="oral", duration_days=5),
        "answer": {"is_valid": False, "drug_class": None},
    },
    {
        "prescription": Prescription(drug_name="Azithromycin", dose_mg=500, frequency="once_daily", route="oral", duration_days=3),
        "answer": {"is_valid": True, "drug_class": "antibiotic_macrolide"},
    },
    {
        "prescription": Prescription(drug_name="Ciprofloxacin", dose_mg=500, frequency="q12h", route="oral", duration_days=7),
        "answer": {"is_valid": True, "drug_class": "antibiotic_fluoroquinolone"},
    },
    {
        "prescription": Prescription(drug_name="Healex", dose_mg=100, frequency="once_daily", route="oral", duration_days=10),
        "answer": {"is_valid": False, "drug_class": None},
    },
]


class Task1:
    task_id = "task1"
    name = "drug_identification"
    difficulty = "easy"
    instruction = (
        "You are a pharmacy validation AI. "
        "Given the prescription below, determine: "
        "(1) Is the drug name valid and recognized? (is_valid: true/false) "
        "(2) What pharmacological class does it belong to? (e.g., antibiotic, NSAID, analgesic_antipyretic) "
        "(3) Set flag_dangerous=true only if the drug name is clearly misspelled or fraudulent. "
        "Respond with a single Action object."
    )

    @staticmethod
    def get_scenario() -> Dict[str, Any]:
        return random.choice(TASK1_SCENARIOS)

    @staticmethod
    def grade(action: Action, scenario: Dict[str, Any]) -> Tuple[float, str]:
        answer = scenario["answer"]
        score = 0.0
        notes = []

        # is_valid: 0.5 weight
        if action.is_valid == answer["is_valid"]:
            score += 0.5
            notes.append("✓ validity correct")
        else:
            notes.append("✗ validity wrong")

        # drug_class: 0.4 weight (only matters if drug is valid)
        if answer["is_valid"]:
            correct_class = answer["drug_class"].lower()
            agent_class = (action.drug_class or "").lower()
            if agent_class == correct_class:
                score += 0.4
                notes.append("✓ drug class correct")
            elif correct_class in agent_class or agent_class in correct_class:
                score += 0.2
                notes.append("~ drug class partially correct")
            else:
                notes.append("✗ drug class wrong")
        else:
            # If invalid drug, class should be None/empty
            if not action.drug_class:
                score += 0.4
                notes.append("✓ correctly left class empty for invalid drug")
            else:
                notes.append("~ class specified for invalid drug (minor deduction)")
                score += 0.1

        # flag_dangerous: 0.1 weight
        if not answer["is_valid"] and action.flag_dangerous:
            score += 0.1
            notes.append("✓ correctly flagged invalid/dangerous drug name")
        elif answer["is_valid"] and not action.flag_dangerous:
            score += 0.1
            notes.append("✓ correctly did not flag valid drug")

        return round(min(0.99, max(0.01, score)), 4), " | ".join(notes)


# ─── TASK 2 — Adult Dosage Verification (Medium) ─────────────────────────────

TASK2_SCENARIOS = [
    {
        "patient": ADULT_PATIENT,
        "prescription": Prescription(drug_name="Ibuprofen", dose_mg=400, frequency="q8h", route="oral", duration_days=5),
        "answer": {"verdict": "safe", "flag_dangerous": False},
    },
    {
        "patient": ADULT_PATIENT,
        "prescription": Prescription(drug_name="Ibuprofen", dose_mg=1200, frequency="q6h", route="oral", duration_days=3),
        "answer": {"verdict": "overdose", "flag_dangerous": True},
    },
    {
        "patient": ADULT_PATIENT,
        "prescription": Prescription(drug_name="Amoxicillin", dose_mg=125, frequency="q8h", route="oral", duration_days=7),
        "answer": {"verdict": "underdose", "flag_dangerous": False},
    },
    {
        "patient": ADULT_FEMALE,
        "prescription": Prescription(drug_name="Paracetamol", dose_mg=1000, frequency="q6h", route="oral", duration_days=3),
        "answer": {"verdict": "safe", "flag_dangerous": False},
    },
    {
        "patient": ADULT_FEMALE,
        "prescription": Prescription(drug_name="Paracetamol", dose_mg=2000, frequency="q4h", route="oral", duration_days=2),
        "answer": {"verdict": "overdose", "flag_dangerous": True},
    },
    {
        "patient": ADULT_PATIENT,
        "prescription": Prescription(drug_name="Ciprofloxacin", dose_mg=500, frequency="q12h", route="oral", duration_days=7),
        "answer": {"verdict": "safe", "flag_dangerous": False},
    },
    {
        "patient": ADULT_PATIENT,
        "prescription": Prescription(drug_name="Metformin", dose_mg=500, frequency="q12h", route="oral", duration_days=30),
        "answer": {"verdict": "safe", "flag_dangerous": False},
    },
    {
        "patient": ADULT_FEMALE,
        "prescription": Prescription(drug_name="Atorvastatin", dose_mg=200, frequency="once_daily", route="oral", duration_days=30),
        "answer": {"verdict": "overdose", "flag_dangerous": True},
    },
]


class Task2:
    task_id = "task2"
    name = "adult_dosage_verification"
    difficulty = "medium"
    instruction = (
        "You are a clinical pharmacist AI. "
        "Given the patient details and prescription below, verify the dose: "
        "(1) verdict: is this dose 'safe', 'overdose', or 'underdose' for this adult patient? "
        "(2) recommended_dose_mg: what is the correct dose range midpoint for this patient? "
        "(3) flag_dangerous: set true if the prescription poses immediate patient harm risk. "
        "(4) reason: brief clinical justification. "
        "is_valid and drug_class must also be set correctly."
    )

    @staticmethod
    def get_scenario() -> Dict[str, Any]:
        return random.choice(TASK2_SCENARIOS)

    @staticmethod
    def grade(action: Action, scenario: Dict[str, Any]) -> Tuple[float, str]:
        answer = scenario["answer"]
        pres = scenario["prescription"]
        score = 0.0
        notes = []

        # is_valid: 0.15
        if action.is_valid:
            score += 0.15
            notes.append("✓ drug validity correct")
        else:
            notes.append("✗ marked valid drug as invalid")

        # verdict: 0.5
        if action.verdict == answer["verdict"]:
            score += 0.5
            notes.append(f"✓ verdict '{answer['verdict']}' correct")
        else:
            notes.append(f"✗ verdict wrong (got '{action.verdict}', expected '{answer['verdict']}')")

        # flag_dangerous: 0.2
        if action.flag_dangerous == answer["flag_dangerous"]:
            score += 0.2
            notes.append("✓ danger flag correct")
        elif answer["flag_dangerous"] and not action.flag_dangerous:
            notes.append("✗ missed dangerous dose — critical miss")
        else:
            score += 0.1
            notes.append("~ false positive on danger flag")

        # recommended_dose: 0.15 (within ±30% of correct range)
        rec = action.recommended_dose_mg
        dose_range = recommended_adult_dose(pres.drug_name.lower())
        if rec and dose_range:
            lo, hi = dose_range
            midpoint = (lo + hi) / 2
            tolerance = midpoint * 0.35
            if abs(rec - midpoint) <= tolerance:
                score += 0.15
                notes.append(f"✓ recommended dose {rec}mg within range")
            else:
                notes.append(f"~ recommended dose {rec}mg outside expected range {lo}-{hi}mg")

        return round(min(0.99, max(0.01, score)), 4), " | ".join(notes)


# ─── TASK 3 — Pediatric + Drug Interaction (Hard) ────────────────────────────

TASK3_SCENARIOS = [
    {
        "patient": CHILD_PATIENT_8,
        "prescriptions": [
            Prescription(drug_name="Aspirin", dose_mg=100, frequency="once_daily", route="oral", duration_days=5),
            Prescription(drug_name="Amoxicillin", dose_mg=500, frequency="q8h", route="oral", duration_days=7),
        ],
        "answer": {
            "flag_dangerous": True,
            "critical_interactions": [],
            "contraindicated_drugs": ["aspirin"],   # Reye's syndrome risk under 12
            "pediatric_verdicts": {"aspirin": "contraindicated", "amoxicillin": "safe"},
        },
    },
    {
        "patient": CHILD_PATIENT_5,
        "prescriptions": [
            Prescription(drug_name="Ibuprofen", dose_mg=200, frequency="q8h", route="oral", duration_days=3),
            Prescription(drug_name="Warfarin", dose_mg=5, frequency="once_daily", route="oral", duration_days=30),
        ],
        "answer": {
            "flag_dangerous": True,
            "critical_interactions": [("warfarin", "ibuprofen")],
            "contraindicated_drugs": [],
            "pediatric_verdicts": {"ibuprofen": "overdose", "warfarin": "overdose"},
        },
    },
    {
        "patient": CHILD_PATIENT_10,
        "prescriptions": [
            Prescription(drug_name="Paracetamol", dose_mg=320, frequency="q6h", route="oral", duration_days=3),
            Prescription(drug_name="Ibuprofen", dose_mg=320, frequency="q8h", route="oral", duration_days=3),
        ],
        "answer": {
            "flag_dangerous": False,
            "critical_interactions": [],
            "contraindicated_drugs": [],
            "pediatric_verdicts": {"paracetamol": "safe", "ibuprofen": "safe"},
        },
    },
    {
        "patient": CHILD_PATIENT_8,
        "prescriptions": [
            Prescription(drug_name="Ciprofloxacin", dose_mg=250, frequency="q12h", route="oral", duration_days=7),
            Prescription(drug_name="Amoxicillin", dose_mg=250, frequency="q8h", route="oral", duration_days=7),
        ],
        "answer": {
            "flag_dangerous": True,
            "critical_interactions": [],
            "contraindicated_drugs": ["ciprofloxacin"],  # <18 contraindicated
            "pediatric_verdicts": {"ciprofloxacin": "contraindicated", "amoxicillin": "underdose"},
        },
    },
]


class Task3:
    task_id = "task3"
    name = "pediatric_interaction_check"
    difficulty = "hard"
    instruction = (
        "You are a senior clinical pharmacist AI reviewing a pediatric prescription. "
        "This is a CHILD patient. You must: "
        "(1) Check every drug for pediatric safety (age/weight-based dosing). "
        "(2) Check all drug pairs for dangerous interactions. "
        "(3) flag_dangerous=true if ANY contraindication, critical interaction, or dangerous dose exists. "
        "(4) interactions_found: list each interaction as 'drug_a + drug_b: severity - description'. "
        "(5) verdict: 'safe_to_dispense' or 'do_not_dispense'. "
        "(6) reason: full clinical justification. "
        "A pharmacist who misses a pediatric contraindication or HIGH interaction causes patient harm."
    )

    @staticmethod
    def get_scenario() -> Dict[str, Any]:
        return random.choice(TASK3_SCENARIOS)

    @staticmethod
    def grade(action: Action, scenario: Dict[str, Any]) -> Tuple[float, str]:
        answer = scenario["answer"]
        prescriptions = scenario["prescriptions"]
        patient = scenario["patient"]
        score = 0.0
        notes = []

        # flag_dangerous: 0.35 — most important signal
        if action.flag_dangerous == answer["flag_dangerous"]:
            score += 0.35
            notes.append("✓ danger flag correct")
        elif answer["flag_dangerous"] and not action.flag_dangerous:
            notes.append("✗ CRITICAL MISS: failed to flag dangerous prescription for child")
        else:
            score += 0.1
            notes.append("~ false positive on danger flag")

        # Interactions found: 0.30
        actual_interaction_pairs = [
            tuple(sorted(pair)) for pair in answer["critical_interactions"]
        ]
        if actual_interaction_pairs:
            interactions_text = " ".join(str(action.interactions_found or [])).lower()
            caught = 0
            for pair in actual_interaction_pairs:
                if pair[0] in interactions_text and pair[1] in interactions_text:
                    caught += 1
            interaction_score = (caught / len(actual_interaction_pairs)) * 0.30
            score += interaction_score
            notes.append(f"interactions: {caught}/{len(actual_interaction_pairs)} caught (+{interaction_score:.2f})")
        else:
            # No interactions — partial credit if agent correctly reports none
            no_interaction_reported = not action.interactions_found or len(action.interactions_found) == 0
            if no_interaction_reported:
                score += 0.30
                notes.append("✓ correctly identified no interactions")
            else:
                score += 0.15
                notes.append("~ reported false interactions (minor)")

        # Contraindicated drugs caught: 0.20
        contra = answer["contraindicated_drugs"]
        if contra:
            agent_text = " ".join([
                action.reason or "",
                " ".join(action.interactions_found or []),
                action.verdict or "",
            ]).lower()
            caught_contra = sum(1 for d in contra if d.lower() in agent_text)
            contra_score = (caught_contra / len(contra)) * 0.20
            score += contra_score
            notes.append(f"contraindications: {caught_contra}/{len(contra)} identified (+{contra_score:.2f})")
        else:
            score += 0.20
            notes.append("✓ no contraindications to miss")

        # Verdict alignment: 0.15
        expected_verdict = "do_not_dispense" if answer["flag_dangerous"] else "safe_to_dispense"
        agent_verdict = (action.verdict or "").lower()
        if expected_verdict in agent_verdict or (
            "not" in agent_verdict and answer["flag_dangerous"]
        ) or (
            "safe" in agent_verdict and not answer["flag_dangerous"]
        ):
            score += 0.15
            notes.append("✓ dispense verdict correct")
        else:
            notes.append("✗ dispense verdict wrong")

        return round(min(0.99, max(0.01, score)), 4), " | ".join(notes)


TASKS = {
    "task1": Task1,
    "task2": Task2,
    "task3": Task3,
}

