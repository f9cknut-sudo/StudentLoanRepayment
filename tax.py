"""
UK tax calculation functions for the Student Loan Plan 2 simulator.

Every function accepts numpy arrays so it can be vectorised across
Monte Carlo iterations. Scalar inputs work too (promoted internally).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

import config as cfg


# ─── Helpers ─────────────────────────────────────────────────────────

def _inflate(base: float, years: int, rate: float = cfg.CPI_RATE) -> float:
    """Grow *base* by *rate* compounded for *years*."""
    return base * (1 + rate) ** years


def _adjust_threshold(
    base: float,
    freeze_until_year: int,
    sim_year: int,
    rate: float = cfg.CPI_RATE,
) -> float:
    """Return a threshold adjusted for freeze then CPI growth.

    Parameters
    ----------
    base : float
        Threshold value at the end of the freeze period.
    freeze_until_year : int
        Tax year (e.g. 2028 means 2028/29) from which CPI applies.
    sim_year : int
        Current simulation tax year.
    rate : float
        Annual CPI growth rate.
    """
    if sim_year < freeze_until_year:
        return base
    years_of_growth = sim_year - freeze_until_year
    return _inflate(base, years_of_growth, rate)


def _adjust_bands(
    bands: List[Tuple[float, float]],
    freeze_until_year: int,
    sim_year: int,
    rate: float = cfg.CPI_RATE,
) -> List[Tuple[float, float]]:
    """CPI-adjust every finite upper limit in a set of tax bands."""
    adjusted = []
    for upper, pct in bands:
        if np.isinf(upper):
            adjusted.append((upper, pct))
        else:
            adjusted.append((_adjust_threshold(upper, freeze_until_year, sim_year, rate), pct))
    return adjusted


# ─── Personal Allowance ─────────────────────────────────────────────

def personal_allowance(gross_income: np.ndarray, sim_year: int = cfg.BASE_TAX_YEAR) -> np.ndarray:
    """Compute personal allowance after the £100k taper.

    For every £2 of income above £100,000 the allowance drops by £1,
    reaching zero at £125,140.

    Parameters
    ----------
    gross_income : array_like
        Annual gross salary/income.
    sim_year : int
        Tax year for threshold adjustment.

    Returns
    -------
    np.ndarray
        Personal allowance for each income value.
    """
    gross_income = np.asarray(gross_income, dtype=float)
    pa = _adjust_threshold(cfg.PERSONAL_ALLOWANCE, cfg.IT_FREEZE_UNTIL, sim_year)
    taper_start = _adjust_threshold(cfg.PA_TAPER_THRESHOLD, cfg.IT_FREEZE_UNTIL, sim_year)

    excess = np.maximum(gross_income - taper_start, 0.0)
    reduction = excess / 2
    return np.maximum(pa - reduction, 0.0)


# ─── Income Tax ──────────────────────────────────────────────────────

def income_tax(
    gross_income: np.ndarray,
    region: str = "england",
    sim_year: int = cfg.BASE_TAX_YEAR,
) -> np.ndarray:
    """Calculate annual income tax.

    Parameters
    ----------
    gross_income : array_like
        Annual gross income.
    region : str
        ``'england'`` (default) or ``'scotland'``.
    sim_year : int
        Simulation tax year for threshold adjustment.

    Returns
    -------
    np.ndarray
        Income tax due for each income value.
    """
    gross_income = np.asarray(gross_income, dtype=float)

    if region == "scotland":
        raw_bands = cfg.INCOME_TAX_BANDS_SCOTLAND
    else:
        raw_bands = cfg.INCOME_TAX_BANDS_ENGLAND

    bands = _adjust_bands(raw_bands, cfg.IT_FREEZE_UNTIL, sim_year)
    pa = personal_allowance(gross_income, sim_year)
    taxable = np.maximum(gross_income - pa, 0.0)

    tax = np.zeros_like(taxable)
    prev_upper = 0.0
    for upper, rate in bands:
        # Band width is measured from the personal-allowance-adjusted zero
        # but the *named* band limits are absolute (include PA).
        # For England: basic band runs from PA to 50,270 of *total* income,
        # so the taxable band width = upper - PA (for the first band) or
        # upper - prev_upper for subsequent bands.  We need the band width
        # in terms of *taxable* income.
        band_floor = max(prev_upper - cfg.PERSONAL_ALLOWANCE, 0.0) if prev_upper > 0 else 0.0
        if np.isinf(upper):
            band_ceil = np.inf
        else:
            pa_base = _adjust_threshold(cfg.PERSONAL_ALLOWANCE, cfg.IT_FREEZE_UNTIL, sim_year)
            band_ceil = upper - pa_base
        band_width = band_ceil - band_floor
        in_band = np.clip(taxable - band_floor, 0.0, band_width)
        tax += in_band * rate
        prev_upper = upper

    return np.round(tax, 2)


# ─── National Insurance ─────────────────────────────────────────────

def national_insurance(
    gross_income: np.ndarray,
    sim_year: int = cfg.BASE_TAX_YEAR,
) -> np.ndarray:
    """Calculate employee Class 1 National Insurance contributions.

    Parameters
    ----------
    gross_income : array_like
        Annual gross income.
    sim_year : int
        Simulation tax year for threshold adjustment.

    Returns
    -------
    np.ndarray
        NI due for each income value.
    """
    gross_income = np.asarray(gross_income, dtype=float)
    bands = _adjust_bands(cfg.NI_BANDS, cfg.NI_FREEZE_UNTIL, sim_year)

    ni = np.zeros_like(gross_income)
    prev_upper = 0.0
    for upper, rate in bands:
        if np.isinf(upper):
            in_band = np.maximum(gross_income - prev_upper, 0.0)
        else:
            adj_upper = upper  # already adjusted by _adjust_bands
            in_band = np.clip(gross_income - prev_upper, 0.0, adj_upper - prev_upper)
        ni += in_band * rate
        prev_upper = upper if not np.isinf(upper) else prev_upper

    return np.round(ni, 2)


# ─── Student Loan ───────────────────────────────────────────────────

def student_loan_threshold(sim_year: int = cfg.BASE_TAX_YEAR, rpi: float = 0.032) -> float:
    """Return the Plan 2 repayment threshold for a given tax year.

    Parameters
    ----------
    sim_year : int
        Tax year (e.g. 2025 = 2025/26).
    rpi : float
        RPI rate used for post-freeze growth.

    Returns
    -------
    float
        Repayment threshold in GBP.
    """
    if sim_year in cfg.SL_PLAN2_THRESHOLDS:
        return float(cfg.SL_PLAN2_THRESHOLDS[sim_year])
    if sim_year < cfg.SL_PLAN2_FREEZE_UNTIL:
        return float(cfg.SL_PLAN2_FREEZE_VALUE)
    # Post-freeze: grow from the frozen value by RPI
    years = sim_year - cfg.SL_PLAN2_FREEZE_UNTIL
    return float(cfg.SL_PLAN2_FREEZE_VALUE * (1 + rpi) ** years)


def student_loan_repayment(
    gross_income: np.ndarray,
    sim_year: int = cfg.BASE_TAX_YEAR,
    rpi: float = 0.032,
) -> np.ndarray:
    """Mandatory annual Plan 2 student loan repayment.

    Parameters
    ----------
    gross_income : array_like
        Annual gross income.
    sim_year : int
        Simulation tax year.
    rpi : float
        RPI rate (used for threshold if post-freeze).

    Returns
    -------
    np.ndarray
        Repayment amount for each income value.
    """
    gross_income = np.asarray(gross_income, dtype=float)
    threshold = student_loan_threshold(sim_year, rpi)
    return np.maximum((gross_income - threshold) * cfg.SL_REPAYMENT_RATE, 0.0)


def student_loan_interest_rate(
    gross_income: np.ndarray,
    rpi: float = 0.032,
    sim_year: int = cfg.BASE_TAX_YEAR,
) -> np.ndarray:
    """Plan 2 interest rate (RPI to RPI+3% sliding scale).

    At or below the repayment threshold: RPI only.
    At or above the upper threshold (£51,245): RPI + 3%.
    Linearly interpolated in between.

    Parameters
    ----------
    gross_income : array_like
        Annual gross income.
    rpi : float
        Current RPI rate.
    sim_year : int
        Simulation tax year.

    Returns
    -------
    np.ndarray
        Interest rate (as a decimal, e.g. 0.062) for each income value.
    """
    gross_income = np.asarray(gross_income, dtype=float)
    threshold = student_loan_threshold(sim_year, rpi)
    upper = cfg.SL_INTEREST_UPPER_THRESHOLD

    fraction = np.clip((gross_income - threshold) / (upper - threshold), 0.0, 1.0)
    return rpi + fraction * 0.03


# ─── Take-Home Pay ──────────────────────────────────────────────────

def take_home_pay(
    gross_income: np.ndarray,
    region: str = "england",
    sim_year: int = cfg.BASE_TAX_YEAR,
    include_student_loan: bool = True,
    rpi: float = 0.032,
) -> np.ndarray:
    """Net annual pay after tax, NI, and optionally student loan.

    Parameters
    ----------
    gross_income : array_like
        Annual gross income.
    region : str
        ``'england'`` or ``'scotland'``.
    sim_year : int
        Simulation tax year.
    include_student_loan : bool
        Whether to deduct student loan repayments.
    rpi : float
        RPI rate for student loan calculations.

    Returns
    -------
    np.ndarray
        Take-home pay for each income value.
    """
    gross_income = np.asarray(gross_income, dtype=float)
    it = income_tax(gross_income, region, sim_year)
    ni = national_insurance(gross_income, sim_year)
    sl = student_loan_repayment(gross_income, sim_year, rpi) if include_student_loan else 0.0
    return gross_income - it - ni - sl


# ─── Marginal Rate Breakdown ────────────────────────────────────────

def marginal_rate_breakdown(
    salary: float,
    region: str = "england",
    sim_year: int = cfg.BASE_TAX_YEAR,
    include_student_loan: bool = True,
    rpi: float = 0.032,
) -> Dict[str, float]:
    """Marginal and effective rate breakdown for a single salary.

    Uses a £1 delta to compute the marginal rate of each component.

    Parameters
    ----------
    salary : float
        Annual gross salary.
    region : str
        ``'england'`` or ``'scotland'``.
    sim_year : int
        Simulation tax year.
    include_student_loan : bool
        Include student loan in the breakdown.
    rpi : float
        RPI rate for student loan calculations.

    Returns
    -------
    dict
        Keys: ``'income_tax_pct'``, ``'ni_pct'``, ``'sl_pct'``,
        ``'total_marginal_pct'``, ``'effective_pct'``.
    """
    s = np.array([salary, salary + 1.0])

    it = income_tax(s, region, sim_year)
    ni = national_insurance(s, sim_year)
    sl = student_loan_repayment(s, sim_year, rpi) if include_student_loan else np.zeros(2)

    it_marginal = float(it[1] - it[0])
    ni_marginal = float(ni[1] - ni[0])
    sl_marginal = float(sl[1] - sl[0]) if include_student_loan else 0.0
    total_marginal = it_marginal + ni_marginal + sl_marginal

    total_deductions = float(it[0]) + float(ni[0]) + (float(sl[0]) if include_student_loan else 0.0)
    effective = total_deductions / salary if salary > 0 else 0.0

    return {
        "income_tax_pct": round(it_marginal * 100, 2),
        "ni_pct": round(ni_marginal * 100, 2),
        "sl_pct": round(sl_marginal * 100, 2),
        "total_marginal_pct": round(total_marginal * 100, 2),
        "effective_pct": round(effective * 100, 2),
    }


# ─── Investment Tax (GIA) ───────────────────────────────────────────

def investment_tax(
    gains: np.ndarray,
    dividends: np.ndarray,
    taxable_income: np.ndarray,
    in_isa: bool = False,
    sim_year: int = cfg.BASE_TAX_YEAR,
) -> np.ndarray:
    """Tax on investment gains and dividends held in a GIA.

    Returns 0 for ISA wrapper. For GIA, applies CGT on gains above
    the annual exempt amount and dividend tax above the allowance,
    at the rate determined by the investor's income tax band.

    Parameters
    ----------
    gains : array_like
        Realised capital gains in the year.
    dividends : array_like
        Dividend income in the year.
    taxable_income : array_like
        Employment/other taxable income (to determine rate band).
    in_isa : bool
        If True, returns zero (ISA is tax-free).
    sim_year : int
        Simulation tax year.

    Returns
    -------
    np.ndarray
        Total investment tax for each element.
    """
    if in_isa:
        return np.zeros_like(np.asarray(gains, dtype=float))

    gains = np.asarray(gains, dtype=float)
    dividends = np.asarray(dividends, dtype=float)
    taxable_income = np.asarray(taxable_income, dtype=float)

    basic_limit = _adjust_threshold(cfg.BASIC_RATE_LIMIT, cfg.IT_FREEZE_UNTIL, sim_year)
    additional_limit = _adjust_threshold(cfg.ADDITIONAL_RATE_LIMIT, cfg.IT_FREEZE_UNTIL, sim_year)

    # CGT
    taxable_gains = np.maximum(gains - cfg.CGT_ANNUAL_EXEMPT, 0.0)
    cgt_rate = np.where(taxable_income <= basic_limit, cfg.CGT_BASIC_RATE, cfg.CGT_HIGHER_RATE)
    cgt = taxable_gains * cgt_rate

    # Dividends
    taxable_divs = np.maximum(dividends - cfg.DIVIDEND_ALLOWANCE, 0.0)
    div_rate = np.where(
        taxable_income <= basic_limit,
        cfg.DIVIDEND_BASIC_RATE,
        np.where(taxable_income <= additional_limit, cfg.DIVIDEND_HIGHER_RATE, cfg.DIVIDEND_ADDITIONAL_RATE),
    )
    div_tax = taxable_divs * div_rate

    return np.round(cgt + div_tax, 2)


# ─── Tests ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests_passed = 0
    tests_failed = 0

    def check(name: str, actual: float, expected: float, tol: float = 1.0) -> None:
        global tests_passed, tests_failed
        passed = abs(actual - expected) <= tol
        status = "PASS" if passed else "FAIL"
        if passed:
            tests_passed += 1
        else:
            tests_failed += 1
        print(f"  [{status}] {name}: expected {expected}, got {actual:.2f}")

    print("=== Income Tax (England) ===")
    check("IT on £35k", float(income_tax(np.array([35_000.0]))[0]), 4_486.0)
    check("IT on £60k", float(income_tax(np.array([60_000.0]))[0]), 11_432.0)

    # £110k — PA taper effect
    it_110k = float(income_tax(np.array([110_000.0]))[0])
    # Without taper PA would be 12,570 → tax ~31,460.  With taper PA = 12570 - (110000-100000)/2 = 7570
    # so extra tax on the lost 5000 of PA at 40% = 2000 more.
    print(f"  [INFO] IT on £110k = £{it_110k:,.2f} (taper adds ~£2k over no-taper)")
    check("IT on £110k (taper)", it_110k, 33_432.0, tol=5.0)

    print("\n=== National Insurance ===")
    check("NI on £35k", float(national_insurance(np.array([35_000.0]))[0]), 1_794.40)

    print("\n=== Student Loan Plan 2 ===")
    check("SL repayment on £35k", float(student_loan_repayment(np.array([35_000.0]))[0]), 587.70)
    check("SL repayment on £60k", float(student_loan_repayment(np.array([60_000.0]))[0]), 2_837.70)

    print("\n=== Student Loan Interest Rate ===")
    sl_ir_35k = float(student_loan_interest_rate(np.array([35_000.0]), rpi=0.032)[0])
    sl_ir_60k = float(student_loan_interest_rate(np.array([60_000.0]), rpi=0.032)[0])
    check("SL interest at £35k (RPI=3.2%)", sl_ir_35k * 100, 3.2 + 0.86, tol=0.5)
    check("SL interest at £60k (RPI=3.2%)", sl_ir_60k * 100, 6.2, tol=0.1)

    print("\n=== Marginal Rates ===")
    mr_35k = marginal_rate_breakdown(35_000.0)
    mr_60k = marginal_rate_breakdown(60_000.0)
    mr_110k = marginal_rate_breakdown(110_000.0)
    check("Marginal at £35k", mr_35k["total_marginal_pct"], 37.0)
    check("Marginal at £60k", mr_60k["total_marginal_pct"], 51.0)
    check("Marginal at £110k", mr_110k["total_marginal_pct"], 71.0)

    print(f"\n{'='*50}")
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
