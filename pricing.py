"""
Deterministic premium calculator — the LLM must NEVER set financial fields.

Usage:
    from pricing import PremiumCalculator
    result = PremiumCalculator.calculate(product, client_data_dict, underwriting_data_dict)
    # result.annual_premium, result.monthly_premium, result.deductible
"""
import math
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_PREMIUM = 500

MAX_PREMIUM: dict[str, int] = {
    "general_liability":      500_000,
    "professional_liability": 250_000,
    "commercial_auto":        150_000,
    "workers_comp":           500_000,
}

# ---------------------------------------------------------------------------
# Industry risk multipliers — keyed by exact INDUSTRY_TYPES values from main.py
# ---------------------------------------------------------------------------

INDUSTRY_RISK_MULTIPLIER: dict[str, float] = {
    # High risk — 2.0–2.5x
    "construction_general":               2.5,
    "roofing_and_siding":                 2.5,
    "freight_and_trucking":               2.0,
    "electrical_contracting":             2.2,
    "hvac_services":                      2.0,
    "plumbing_services":                  2.0,
    "flooring_and_tile":                  2.0,
    "moving_and_storage":                 2.0,
    "pest_control":                       2.0,
    "security_services":                  2.0,
    "landscape_and_lawn_care":            1.8,
    # Medium risk — 1.3–1.5x
    "restaurant_and_bar":                 1.5,
    "retail_general":                     1.3,
    "food_truck_and_catering":            1.4,
    "bakery_and_food_production":         1.4,
    "auto_repair_and_maintenance":        1.5,
    "health_and_medical":                 1.5,
    "dental_practice":                    1.4,
    "veterinary_services":                1.4,
    "manufacturing_light":                1.5,
    "property_management":                1.3,
    "cleaning_services":                  1.3,
    "janitorial_and_commercial_cleaning": 1.3,
    "fitness_and_wellness":               1.3,
    "entertainment_and_events":           1.4,
    "hospitality_and_hotels":             1.4,
    "beauty_and_personal_care":           1.2,
    "pet_services":                       1.2,
    "printing_and_signage":               1.2,
    "clothing_and_apparel":               1.2,
    "jewelry_and_accessories":            1.2,
    "funeral_services":                   1.3,
    "real_estate":                        1.3,
    "staffing_and_hr":                    1.3,
    "agriculture_and_farming":            1.5,
    # Low risk — 0.8–0.9x
    "consulting_business":                0.8,
    "consulting_it":                      0.8,
    "consulting_management":              0.8,
    "accounting_and_bookkeeping":         0.8,
    "graphic_design_and_marketing":       0.8,
    "photography_and_videography":        0.8,
    "education_and_tutoring":             0.8,
    "media_and_publishing":               0.8,
    "nonprofit":                          0.8,
    "legal_services":                     0.9,
    "financial_services":                 0.9,
    "insurance_agency":                   0.9,
    "it_services_and_msp":                0.9,
    "e_commerce":                         0.9,
}

# Coverage-limit factor for professional_liability
COVERAGE_LIMIT_FACTOR: dict[str, float] = {
    "$500k": 0.75,
    "$1m":   1.0,
    "$2m":   1.5,
    "$5m":   2.5,
}

# Workers comp job class multipliers
WORKERS_COMP_JOB_CLASS_MULTIPLIER: dict[str, float] = {
    "office":        0.5,
    "clerical":      0.5,
    "retail":        1.0,
    "field":         1.5,
    "construction":  2.5,
    "roofing":       3.0,
    "trucking":      2.0,
    "electrical":    2.2,
    "hvac":          2.0,
    "plumbing":      2.0,
    "manufacturing": 1.8,
    "restaurant":    1.2,
}

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PremiumResult:
    annual_premium:  int
    monthly_premium: int
    deductible:      int

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, product: str) -> int:
    return round(max(MIN_PREMIUM, min(value, MAX_PREMIUM[product])))


def _monthly(annual: int) -> int:
    return math.ceil(annual / 12)


def _parse_payroll(payroll_str: str) -> float:
    """
    Parse free-text payroll strings into a float dollar amount.
    Handles: "$250,000", "250k", "1.5M", "$1.2 million", "250000"
    Returns 0.0 if unparseable (floors to MIN_PREMIUM).
    """
    s = payroll_str.lower().replace(",", "").replace("$", "").strip()
    s = re.sub(r"\s*million", "m", s)
    s = re.sub(r"\s*thousand", "k", s)
    m = re.match(r"^([\d.]+)\s*([mk]?)", s)
    if not m:
        return 0.0
    value = float(m.group(1))
    suffix = m.group(2)
    if suffix == "m":
        value *= 1_000_000
    elif suffix == "k":
        value *= 1_000
    return value


def _industry_multiplier(industry_type: str) -> float:
    return INDUSTRY_RISK_MULTIPLIER.get(industry_type, 1.0)


def _coverage_limit_factor(coverage_limit: str) -> float:
    key = coverage_limit.strip().lower().replace(" ", "")
    return COVERAGE_LIMIT_FACTOR.get(key, 1.0)


def _job_class_multiplier(job_classes: str) -> float:
    text = job_classes.lower()
    found = [v for k, v in WORKERS_COMP_JOB_CLASS_MULTIPLIER.items() if k in text]
    return max(found) if found else 1.0

# ---------------------------------------------------------------------------
# Per-product calculators
# ---------------------------------------------------------------------------

def _general_liability(client: dict, uw: dict) -> PremiumResult:
    revenue       = max(int(client.get("annual_revenue", 0)), 0)
    prior_claims  = max(int(uw.get("prior_claims", 0)), 0)
    years         = max(int(client.get("years_in_business", 0)), 0)
    industry_type = client.get("industry_type", "")

    base = revenue * 0.012
    base *= _industry_multiplier(industry_type)
    base *= (1 + 0.15 * prior_claims)
    discount = min(0.05 * (years // 3), 0.20)
    base *= (1 - discount)

    annual     = _clamp(base, "general_liability")
    deductible = max(round(annual * 0.02), 250)
    return PremiumResult(annual, _monthly(annual), deductible)


def _professional_liability(client: dict, uw: dict) -> PremiumResult:
    revenue        = max(int(client.get("annual_revenue", 0)), 0)
    prior_claims   = max(int(uw.get("prior_claims", 0)), 0)
    coverage_limit = uw.get("coverage_limit", "$1M")

    base = revenue * 0.020
    base *= _coverage_limit_factor(coverage_limit)
    base *= (1 + 0.20 * prior_claims)

    annual     = _clamp(base, "professional_liability")
    deductible = max(round(annual * 0.05), 500)
    return PremiumResult(annual, _monthly(annual), deductible)


def _commercial_auto(client: dict, uw: dict) -> PremiumResult:
    vehicles  = max(int(uw.get("vehicle_count", 1)), 1)
    young     = bool(uw.get("young_drivers", False))
    incidents = max(int(uw.get("incidents", 0)), 0)

    base = 1_800 * vehicles
    if young:
        base += 400 * vehicles
    base += 600 * incidents

    annual     = _clamp(base, "commercial_auto")
    deductible = max(1_000 * vehicles, 1_000)
    return PremiumResult(annual, _monthly(annual), deductible)


def _workers_comp(client: dict, uw: dict) -> PremiumResult:
    payroll_str    = uw.get("payroll", "0")
    prior_injuries = max(int(uw.get("prior_injuries", 0)), 0)
    job_classes    = uw.get("job_classes", "")

    payroll = _parse_payroll(payroll_str)
    base = (payroll / 100) * 2.50
    base *= _job_class_multiplier(job_classes)
    base *= (1 + 0.25 * prior_injuries)

    annual = _clamp(base, "workers_comp")
    return PremiumResult(annual, _monthly(annual), deductible=0)

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class PremiumCalculator:
    _CALCULATORS = {
        "general_liability":      _general_liability,
        "professional_liability": _professional_liability,
        "commercial_auto":        _commercial_auto,
        "workers_comp":           _workers_comp,
    }

    @classmethod
    def calculate(cls, product: str, client_data: dict, underwriting_data: dict) -> PremiumResult:
        fn = cls._CALCULATORS.get(product)
        if fn is None:
            raise ValueError(
                f"PremiumCalculator: unknown product '{product}'. "
                f"Must be one of {list(cls._CALCULATORS)}"
            )
        return fn(client_data, underwriting_data)
