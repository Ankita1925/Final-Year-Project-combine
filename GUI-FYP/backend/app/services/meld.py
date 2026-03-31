# services/meld.py
#
# Single source of truth for all liver severity scoring:
#   - MELD score (with optional MELD-Na sodium correction)
#   - MELD risk classification + transplant threshold
#   - Child-Pugh score + classification

import math
import logging

logger = logging.getLogger(__name__)


# -----------------------------------------------
# MELD Score
# -----------------------------------------------

def calculate_meld(
    bilirubin: float,
    inr: float,
    creatinine: float,
    sodium: float | None = None
) -> int | None:
    """
    Calculate MELD or MELD-Na score.

    Standard MELD formula (UNOS):
        MELD = 3.78 * ln(Bilirubin) + 11.2 * ln(INR) + 9.57 * ln(Creatinine) + 6.43

    MELD-Na correction (if sodium provided):
        MELD-Na = MELD + 1.32 * (137 - Na) - (0.033 * MELD * (137 - Na))
        Sodium is clamped to [125, 137] per UNOS guidelines.

    All inputs floored at 1.0 before log to prevent log(0) crashes.
    Creatinine capped at 4.0 (UNOS dialysis rule).

    Returns:
        Rounded integer MELD score, or None if required values are missing.
    """
    if bilirubin is None or inr is None or creatinine is None:
        logger.warning("MELD calculation skipped: missing required values.")
        return None

    # Apply UNOS rules
    bilirubin  = max(bilirubin, 1.0)
    inr        = max(inr, 1.0)
    creatinine = min(max(creatinine, 1.0), 4.0)   # cap at 4.0 for dialysis patients

    meld = (
        3.78  * math.log(bilirubin) +
        11.2  * math.log(inr) +
        9.57  * math.log(creatinine) +
        6.43
    )

    # MELD-Na correction
    if sodium is not None:
        sodium = max(min(sodium, 137.0), 125.0)    # clamp per UNOS
        meld = meld + 1.32 * (137 - sodium) - (0.033 * meld * (137 - sodium))

    return round(meld)


# -----------------------------------------------
# MELD Risk Classification
# -----------------------------------------------

def classify_meld(meld_score: int | None) -> dict | None:
    """
    Classify MELD score into risk bands and determine transplant need.

    Transplant threshold: MELD ≥ 15 (UNOS clinical standard for listing).

    Returns a dict with:
        risk_level         : str   — "Low" / "Moderate" / "High" / "Critical"
        transplant_required: bool  — True if MELD ≥ 15
        mortality_90day    : str   — approximate 90-day mortality label
        description        : str   — plain-English explanation for the report
    """
    if meld_score is None:
        return None

    if meld_score < 10:
        return {
            "risk_level":          "Low",
            "transplant_required": False,
            "mortality_90day":     "< 2%",
            "description":         "Stable cirrhosis. Regular outpatient follow-up advised."
        }
    elif meld_score < 15:
        return {
            "risk_level":          "Moderate",
            "transplant_required": False,
            "mortality_90day":     "~6%",
            "description":         "Clinically significant liver dysfunction. Close monitoring required."
        }
    elif meld_score < 20:
        return {
            "risk_level":          "High",
            "transplant_required": True,
            "mortality_90day":     "~20%",
            "description":         "Transplant listing evaluation recommended. Hospital admission may be needed."
        }
    elif meld_score < 30:
        return {
            "risk_level":          "Very High",
            "transplant_required": True,
            "mortality_90day":     "~40%",
            "description":         "Urgent transplant evaluation required. High risk of decompensation."
        }
    else:
        return {
            "risk_level":          "Critical",
            "transplant_required": True,
            "mortality_90day":     "> 70%",
            "description":         "Critical. Immediate ICU-level care and emergency transplant evaluation."
        }


# -----------------------------------------------
# Child-Pugh Score
# -----------------------------------------------

def calculate_child_pugh(
    bilirubin: float,
    albumin: float,
    inr: float,
    ascites: int,
    encephalopathy: int
) -> dict | None:
    """
    Calculate Child-Pugh score and classify cirrhosis severity.

    Scoring:
        Bilirubin (mg/dL): < 2 → 1pt | 2–3 → 2pt | > 3 → 3pt
        Albumin (g/dL):    > 3.5 → 1pt | 2.8–3.5 → 2pt | < 2.8 → 3pt
        INR:               < 1.7 → 1pt | 1.7–2.3 → 2pt | > 2.3 → 3pt
        Ascites:           0=None → 1pt | 1=Mild → 2pt | 2=Severe → 3pt
        Encephalopathy:    0=None → 1pt | 1=Grade1-2 → 2pt | 2=Grade3-4 → 3pt

    Classification:
        5–6  → Child-Pugh A (Well Compensated)       1-year survival ~100%
        7–9  → Child-Pugh B (Significant Compromise)  1-year survival ~80%
        10–15 → Child-Pugh C (Decompensated)           1-year survival ~45%

    Returns:
        Dict with score, classification, survival estimate, and description.
        None if any required value is missing.
    """
    if None in (bilirubin, albumin, inr, ascites, encephalopathy):
        logger.warning("Child-Pugh skipped: one or more required values are None.")
        return None

    score = 0

    # Bilirubin
    if bilirubin < 2:
        score += 1
    elif bilirubin <= 3:
        score += 2
    else:
        score += 3

    # Albumin
    if albumin > 3.5:
        score += 1
    elif albumin >= 2.8:
        score += 2
    else:
        score += 3

    # INR
    if inr < 1.7:
        score += 1
    elif inr <= 2.3:
        score += 2
    else:
        score += 3

    # Ascites: patient input is 0/1/2, maps to 1/2/3 points
    score += (ascites + 1)

    # Encephalopathy: same mapping
    score += (encephalopathy + 1)

    # Classify
    if score <= 6:
        classification  = "Child-Pugh A"
        severity        = "Well Compensated"
        survival_1yr    = "~100%"
        survival_2yr    = "~85%"
        description     = "Well-compensated cirrhosis. Medical management appropriate."
    elif score <= 9:
        classification  = "Child-Pugh B"
        severity        = "Significant Functional Compromise"
        survival_1yr    = "~80%"
        survival_2yr    = "~60%"
        description     = "Significant liver dysfunction. Specialist review and optimisation required."
    else:
        classification  = "Child-Pugh C"
        severity        = "Decompensated"
        survival_1yr    = "~45%"
        survival_2yr    = "~35%"
        description     = "Decompensated cirrhosis. Transplant evaluation strongly recommended."

    return {
        "score":          score,
        "classification": classification,
        "severity":       severity,
        "survival_1yr":   survival_1yr,
        "survival_2yr":   survival_2yr,
        "description":    description,
    }