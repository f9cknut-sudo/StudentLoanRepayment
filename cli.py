"""
CLI interface and shared display-data computation for the
Student Loan Plan 2 overpay-vs-invest simulator.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np

import config as cfg
import tax
from simulation import (
    BreakevenResult,
    SimInputs,
    SimResults,
    breakeven_table,
    parameter_sweep,
    run_simulation,
)
import report


# ═══════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════

def fmt(val: float, decimals: int = 0) -> str:
    """Format number as £X,XXX."""
    if decimals > 0:
        return f"\u00a3{val:,.{decimals}f}"
    return f"\u00a3{val:,.0f}"


def pct(val: float, decimals: int = 1) -> str:
    return f"{val:.{decimals}f}%"


# ═══════════════════════════════════════════════════════════════════
# Input collection (CLI)
# ═══════════════════════════════════════════════════════════════════

def _strip_currency(s: str) -> str:
    """Remove currency symbols, commas, spaces."""
    return s.replace("\u00a3", "").replace(",", "").replace(" ", "")


def _prompt_float(
    label: str,
    default: Any,
    min_val: float | None = None,
    max_val: float | None = None,
    currency: bool = False,
) -> float:
    while True:
        raw = input(f"  {label} [{default}]: ").strip()
        if not raw:
            return float(_strip_currency(str(default))) if currency else float(default)
        try:
            val = float(_strip_currency(raw) if currency else raw.replace("%", ""))
            if min_val is not None and val < min_val:
                print(f"    Must be at least {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"    Must be at most {max_val}")
                continue
            return val
        except ValueError:
            print("    Invalid number, try again.")


def _prompt_int(
    label: str,
    default: int,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    while True:
        raw = input(f"  {label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(float(_strip_currency(raw)))
            if min_val is not None and val < min_val:
                print(f"    Must be at least {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"    Must be at most {max_val}")
                continue
            return val
        except ValueError:
            print("    Invalid number, try again.")


def _prompt_choice(label: str, options: list[str], default: str) -> str:
    opts = "/".join(options)
    while True:
        raw = input(f"  {label} ({opts}) [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in options:
            return raw
        print(f"    Choose from: {opts}")


def collect_inputs() -> SimInputs:
    """Prompt the user for all simulation parameters."""
    print("\n  Enter your details (press Enter for defaults):\n")

    age = _prompt_int("Current age", 25, 18, 65)
    loan = _prompt_float("Starting loan balance", "\u00a345,000", 25_000, currency=True)
    salary = _prompt_float("Current annual gross salary", "\u00a335,000", 0, currency=True)
    growth = _prompt_float("Expected salary growth %/yr", 3.5, 0, 20) / 100
    years = _prompt_int("Years since first repayment", 1, 0, 29)
    region = _prompt_choice("Region", ["england", "scotland"], "england")
    overpay = _prompt_float("Monthly overpayment amount", "\u00a3200", 0, currency=True)
    isa_choice = _prompt_choice("Invest in ISA?", ["yes", "no"], "yes")
    inv_ret = _prompt_float("Expected annual investment return %", 7.0, 0, 30) / 100
    inv_vol = _prompt_float("Return volatility / std dev %", 15.0, 0, 50) / 100
    rpi = _prompt_float("Average RPI %", 3.2, 0, 15) / 100
    rpi_vol = _prompt_float("RPI volatility %", 1.5, 0, 10) / 100
    n_iter = _prompt_int("Number of iterations", 10_000, 100, 100_000)

    return SimInputs(
        loan_balance=loan,
        salary=salary,
        salary_growth_mean=growth,
        age=age,
        years_since_first_repayment=years,
        region=region,
        monthly_overpayment=overpay,
        isa=isa_choice == "yes",
        inv_return_mean=inv_ret,
        inv_return_std=inv_vol,
        rpi_mean=rpi,
        rpi_std=rpi_vol,
        n_iterations=n_iter,
    )


# ═══════════════════════════════════════════════════════════════════
# Shared display-data computation (used by CLI and web app)
# ═══════════════════════════════════════════════════════════════════

def _lump_sum_hypothetical(
    loan_balance: float,
    annual_mandatory: float,
    expected_return: float,
    remaining_years: int,
) -> Dict[str, Any]:
    """Compute a deterministic lump sum payoff vs invest comparison.

    Pay off now:  spend £loan_balance, invest freed mandatory payments monthly.
    Invest lump:  invest £loan_balance today, keep paying mandatory, loan written off.

    Uses the stated expected return (no stochastic modelling) — clearly a
    simplified hypothetical.
    """
    r = expected_return
    T = remaining_years

    # Pay off: freed mandatory payments invested as an annuity
    # FV of annuity = PMT * [((1+r)^T - 1) / r]
    if r > 0:
        payoff_fv = annual_mandatory * (((1 + r) ** T - 1) / r)
    else:
        payoff_fv = annual_mandatory * T

    # Invest the lump sum: grows for T years
    invest_fv = loan_balance * (1 + r) ** T

    # Total mandatory payments over the period (cost in invest scenario,
    # saved in payoff scenario)
    total_mandatory_paid = annual_mandatory * T

    if invest_fv >= payoff_fv:
        ls_winner = "invest"
        ls_advantage = invest_fv - payoff_fv
    else:
        ls_winner = "payoff"
        ls_advantage = payoff_fv - invest_fv

    ls_loser = min(invest_fv, payoff_fv)
    ls_adv_pct = (ls_advantage / ls_loser * 100) if ls_loser > 0 else 0.0

    return {
        "ls_lump_sum": loan_balance,
        "ls_payoff_fv": payoff_fv,
        "ls_invest_fv": invest_fv,
        "ls_freed_annual": annual_mandatory,
        "ls_freed_monthly": annual_mandatory / 12,
        "ls_total_mandatory_saved": total_mandatory_paid,
        "ls_winner": ls_winner,
        "ls_advantage": ls_advantage,
        "ls_adv_pct": ls_adv_pct,
        "ls_years": T,
        "ls_return": expected_return,
    }


def _invest_favorable_below(be: BreakevenResult) -> Optional[float]:
    """Return the breakeven threshold below which investing is favorable.

    If overpaying never wins, returns None (investing is always favorable).
    Otherwise returns the breakeven_amount — investing wins below that level.
    """
    return be.breakeven_amount


def _invest_favorable_max_tested(be: BreakevenResult) -> float:
    """Return the highest tested overpayment where investing still wins."""
    max_invest = 0.0
    for r in be.rows:
        if r.winner == "invest" and r.monthly_overpayment > max_invest:
            max_invest = r.monthly_overpayment
    return max_invest


def compute_display_data(
    inputs: SimInputs,
    results: SimResults,
    be: BreakevenResult,
) -> Dict[str, Any]:
    """Extract every metric needed for the output sections."""
    T = inputs.remaining_years
    sal_arr = np.array([inputs.salary])

    # ── Section 1: You Now ──────────────────────────────────────
    annual_take_home = float(tax.take_home_pay(sal_arr, inputs.region, rpi=inputs.rpi_mean)[0])
    marginal = tax.marginal_rate_breakdown(inputs.salary, inputs.region, rpi=inputs.rpi_mean)
    annual_mandatory = float(tax.student_loan_repayment(sal_arr, rpi=inputs.rpi_mean)[0])
    monthly_mandatory = annual_mandatory / 12
    sl_rate = float(tax.student_loan_interest_rate(sal_arr, inputs.rpi_mean)[0])

    # ── Section 2: Option A ─────────────────────────────────────
    pct_cleared = float(results.a_loan_cleared.mean() * 100)
    cleared_years = results.a_year_loan_cleared[~np.isnan(results.a_year_loan_cleared)]

    if len(cleared_years) > 0 and pct_cleared >= 50:
        med_year = float(np.median(cleared_years))
        med_age = inputs.age + med_year
        years_investing = T - med_year
    else:
        med_year = None
        med_age = None
        years_investing = 0.0

    a_nw = float(np.median(results.a_net_worth[T]))
    a_paid = float(np.median(results.a_total_repaid[T]))

    # ── Section 3: Option B ─────────────────────────────────────
    b_nw = float(np.median(results.b_net_worth[T]))
    b_paid = float(np.median(results.b_total_repaid[T]))
    b_forgiven = float(np.median(results.b_amount_written_off))

    # ── Section 4: Verdict ──────────────────────────────────────
    loser_nw = min(a_nw, b_nw)
    if b_nw >= a_nw:
        winner = "invest"
        adv_abs = b_nw - a_nw
    else:
        winner = "overpay"
        adv_abs = a_nw - b_nw
    adv_pct = (adv_abs / loser_nw * 100) if loser_nw > 0 else 0.0

    invest_wins_pct = float((results.b_net_worth[T] > results.a_net_worth[T]).mean() * 100)

    return {
        # Inputs echo
        "age": inputs.age,
        "salary": inputs.salary,
        "loan_balance": inputs.loan_balance,
        "region": inputs.region,
        "overpayment": inputs.monthly_overpayment,
        "isa": inputs.isa,
        "inv_return": inputs.inv_return_mean,
        "remaining_years": T,
        "write_off_age": inputs.age + T,
        # Section 1
        "annual_take_home": annual_take_home,
        "monthly_take_home": annual_take_home / 12,
        "marginal": marginal,
        "annual_mandatory": annual_mandatory,
        "monthly_mandatory": monthly_mandatory,
        "sl_rate": sl_rate,
        # Section 2
        "a_pct_cleared": pct_cleared,
        "a_med_year": med_year,
        "a_med_age": med_age,
        "a_years_investing": years_investing,
        "a_freed_overpayment": inputs.monthly_overpayment,
        "a_freed_mandatory": monthly_mandatory,
        "a_monthly_invest": inputs.monthly_overpayment + monthly_mandatory,
        "a_annual_reinvested": (inputs.monthly_overpayment + monthly_mandatory) * 12,
        "a_total_reinvested": (inputs.monthly_overpayment + monthly_mandatory) * 12 * years_investing,
        "a_nw": a_nw,
        "a_paid": a_paid,
        # Section 3
        "b_monthly_invest": inputs.monthly_overpayment,
        "b_nw": b_nw,
        "b_paid": b_paid,
        "b_forgiven": b_forgiven,
        # Section 4
        "winner": winner,
        "adv_abs": adv_abs,
        "adv_pct": adv_pct,
        "invest_wins_pct": invest_wins_pct,
        # Section 5
        "breakeven": be,
        # Invest-favorable threshold: highest overpayment where investing still wins
        "invest_favorable_below": _invest_favorable_below(be),
        "invest_favorable_max_tested": _invest_favorable_max_tested(be),
        # Lump sum hypothetical
        **_lump_sum_hypothetical(inputs.loan_balance, annual_mandatory,
                                 inputs.inv_return_mean, T),
    }


def generate_verdict_text(d: Dict[str, Any]) -> str:
    """Build a 2-3 sentence plain-English verdict."""
    overpay = fmt(d["overpayment"])
    adv = fmt(d["adv_abs"])
    adv_p = pct(d["adv_pct"])
    T = d["remaining_years"]
    sl = pct(d["sl_rate"] * 100, 1)
    ret = pct(d["inv_return"] * 100, 1)

    if d["winner"] == "invest":
        if d["a_pct_cleared"] < 60:
            return (
                f"Investing {overpay}/mo wins by {adv} ({adv_p}). "
                f"In only {d['a_pct_cleared']:.0f}% of scenarios do your "
                f"overpayments clear the loan before write-off at age "
                f"{d['write_off_age']} \u2014 the rest is money paid towards "
                f"a debt that gets forgiven anyway. With {T} years of "
                f"compound growth at ~{ret}, your investment pot far "
                f"outpaces the {sl} loan interest you\u2019d save."
            )
        years_inv = d["a_years_investing"]
        return (
            f"Investing {overpay}/mo wins by {adv} ({adv_p}). "
            f"Even though overpayments would clear the loan by age "
            f"{d['a_med_age']:.0f} (freeing {fmt(d['a_monthly_invest'])}/mo "
            f"for {years_inv:.0f} years), starting to invest from day one "
            f"for the full {T} years at ~{ret} outweighs the {sl} "
            f"loan interest saved."
        )
    else:
        years_inv = d["a_years_investing"]
        return (
            f"Overpaying {overpay}/mo wins by {adv} ({adv_p}). "
            f"Clearing your loan by age {d['a_med_age']:.0f} "
            f"({d['a_pct_cleared']:.0f}% of scenarios) frees up "
            f"{fmt(d['a_monthly_invest'])}/mo to invest for "
            f"{years_inv:.0f} years. The savings on your {sl} "
            f"loan interest plus {years_inv:.0f} years of freed-up "
            f"compounding beat investing {overpay}/mo from day one."
        )


# ═══════════════════════════════════════════════════════════════════
# Box-drawing CLI output
# ═══════════════════════════════════════════════════════════════════

W = 78  # box width (characters)


def _box_top(title: str) -> str:
    inner = W - 2
    return (
        f"\u2554{'\u2550' * inner}\u2557\n"
        f"\u2551  {title:<{inner - 2}}\u2551\n"
        f"\u2560{'\u2550' * inner}\u2563"
    )


def _box_line(text: str = "") -> str:
    inner = W - 4
    if len(text) > inner:
        text = text[:inner]
    return f"\u2551  {text:<{inner}}\u2551"


def _box_row(label: str, value: str, lw: int = 38) -> str:
    return _box_line(f"{label:<{lw}}{value}")


def _box_bottom() -> str:
    return f"\u255a{'\u2550' * (W - 2)}\u255d"


def _box_sep() -> str:
    return f"\u2560{'\u2550' * (W - 2)}\u2563"


def _print_section(title: str, rows: List[str]) -> None:
    """Print a titled box with content rows."""
    print(_box_top(title))
    for r in rows:
        print(r)
    print(_box_bottom())
    print()


# ═══════════════════════════════════════════════════════════════════
# CLI Section Printers
# ═══════════════════════════════════════════════════════════════════

def _print_section_1(d: Dict[str, Any]) -> None:
    m = d["marginal"]
    marginal_breakdown = (
        f"{pct(m['income_tax_pct'])} IT + "
        f"{pct(m['ni_pct'])} NI + "
        f"{pct(m['sl_pct'])} SL"
    )
    rows = [
        _box_row("Age", str(d["age"])),
        _box_row("Salary", fmt(d["salary"])),
        _box_row("Loan balance", fmt(d["loan_balance"])),
        _box_line(),
        _box_row("Take-home pay (annual)", fmt(d["annual_take_home"])),
        _box_row("Take-home pay (monthly)", fmt(d["monthly_take_home"])),
        _box_row("Marginal rate", pct(m["total_marginal_pct"])),
        _box_row("  Breakdown", marginal_breakdown),
        _box_line(),
        _box_row("Monthly mandatory SL cost", fmt(d["monthly_mandatory"])),
        _box_row("Loan interest rate", pct(d["sl_rate"] * 100)),
        _box_row("Years until write-off", str(d["remaining_years"])),
        _box_row("Age at write-off", str(d["write_off_age"])),
    ]
    _print_section("YOU NOW", rows)


def _print_section_2(d: Dict[str, Any]) -> None:
    rows = []

    if d["a_med_age"] is not None:
        rows.append(_box_row("Loan clears at age (median)", f"{d['a_med_age']:.0f}"))
        rows.append(_box_row("Loan clears in year (median)", f"{d['a_med_year']:.0f}"))
    else:
        rows.append(_box_row("Loan clears at age (median)", "Does not clear"))

    rows.append(_box_row("Simulations where loan clears", pct(d["a_pct_cleared"])))
    rows.append(_box_line())

    rows.append(_box_row("Monthly invest after clearing", fmt(d["a_monthly_invest"])))
    rows.append(_box_row(
        "  Components",
        f"{fmt(d['a_freed_overpayment'])} overpayment + {fmt(d['a_freed_mandatory'])} mandatory",
    ))

    if d["a_years_investing"] > 0:
        rows.append(_box_row("Years of investing", f"{d['a_years_investing']:.0f}"))

    rows.append(_box_line())
    rows.append(_box_row("Net worth at write-off (median)", fmt(d["a_nw"])))
    rows.append(_box_row("Total lifetime payments to SLC", fmt(d["a_paid"])))

    if d["a_pct_cleared"] < 60:
        rows.append(_box_line())
        warn_pct = 100 - d["a_pct_cleared"]
        rows.append(_box_line(f"WARNING: In {warn_pct:.0f}% of scenarios your"))
        rows.append(_box_line("overpayments don't clear the loan before"))
        rows.append(_box_line("write-off - that money would be wasted on"))
        rows.append(_box_line("a debt that gets forgiven anyway."))

    _print_section(f"OPTION A \u2014 OVERPAY {fmt(d['overpayment'])}/MO", rows)


def _print_section_3(d: Dict[str, Any]) -> None:
    rows = [
        _box_row("Invest per month", fmt(d["b_monthly_invest"])),
        _box_row("Investment duration", f"{d['remaining_years']} years (full term)"),
        _box_row("Loan written off at age", str(d["write_off_age"])),
        _box_row("Amount forgiven for free", fmt(d["b_forgiven"])),
        _box_line(),
        _box_row("Net worth at write-off (median)", fmt(d["b_nw"])),
        _box_row("Total lifetime payments to SLC", fmt(d["b_paid"])),
    ]
    _print_section(f"OPTION B \u2014 INVEST {fmt(d['overpayment'])}/MO INSTEAD", rows)


def _print_section_4(d: Dict[str, Any]) -> None:
    if d["winner"] == "invest":
        winner_label = "INVESTING WINS"
    else:
        winner_label = "OVERPAYING WINS"

    rows = [
        _box_row("Winner", winner_label),
        _box_row("Advantage", f"{fmt(d['adv_abs'])} ({pct(d['adv_pct'])})"),
        _box_line(),
    ]

    # Word-wrap the verdict text
    verdict = generate_verdict_text(d)
    line_len = W - 6
    words = verdict.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 <= line_len:
            line = f"{line} {word}" if line else word
        else:
            rows.append(_box_line(line))
            line = word
    if line:
        rows.append(_box_line(line))

    _print_section("THE VERDICT", rows)


def _print_section_5(d: Dict[str, Any]) -> None:
    be: BreakevenResult = d["breakeven"]

    # Table header
    h1 = f"{'Overpay':>8}  {'Clears':>7}  {'Age':>5}  {'Winner':>8}  {'By':>10}"
    rows = [_box_line(h1), _box_line("\u2500" * (W - 6))]

    for r in be.rows:
        is_breakeven = (
            be.breakeven_amount is not None
            and r.monthly_overpayment == be.breakeven_amount
        )
        marker = " <<" if is_breakeven else ""
        age_str = f"{r.median_age_cleared:.0f}" if r.median_age_cleared is not None else "  -"
        clears_str = f"{r.pct_loan_clears:.0f}%"
        line = (
            f"{fmt(r.monthly_overpayment):>8}  "
            f"{clears_str:>7}  "
            f"{age_str:>5}  "
            f"{r.winner:>8}  "
            f"{fmt(r.advantage):>10}"
            f"{marker}"
        )
        rows.append(_box_line(line))

    rows.append(_box_line())

    # One-liner summary
    if be.breakeven_amount is not None:
        summary = (
            f"You'd need to overpay {fmt(be.breakeven_amount)}/mo+ for "
            f"it to beat investing."
        )
    else:
        summary = (
            "No amount up to \u00a31,000/mo beats investing "
            "\u2014 your loan will be written off."
        )
    rows.append(_box_line(summary))

    rows.append(_box_line())
    rows.append(_box_line("TIP: Salary sacrifice into your pension reduces"))
    rows.append(_box_line("both income tax and student loan repayments,"))
    rows.append(_box_line("effectively cutting your marginal rate."))

    _print_section("WHAT OVERPAYMENT WOULD BE WORTHWHILE?", rows)


def _print_section_6(pdf_path: str | None) -> None:
    rows = []
    if pdf_path:
        rows.append(_box_line(f"PDF report saved to: {pdf_path}"))
    else:
        rows.append(_box_line("Charts available in the web app:"))
        rows.append(_box_line("  python main.py  (opens localhost:5000)"))
    _print_section("CHARTS", rows)


# ═══════════════════════════════════════════════════════════════════
# Main CLI entry point
# ═══════════════════════════════════════════════════════════════════

def run_cli() -> None:
    """Run the full CLI workflow."""
    # Ensure box-drawing characters render on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    print()
    print("=" * W)
    print("  Student Loan Plan 2: Overpay vs Invest Simulator")
    print("=" * W)

    inputs = collect_inputs()

    print(f"\n  Running Monte Carlo simulation ({inputs.n_iterations:,} iterations)...")
    results = run_simulation(inputs)
    print("  Done.")

    print(f"\n  Computing breakeven table...")
    be = breakeven_table(inputs)
    print("  Done.")

    # Compute display data
    d = compute_display_data(inputs, results, be)

    print()
    _print_section_1(d)
    _print_section_2(d)
    _print_section_3(d)
    _print_section_4(d)
    _print_section_5(d)

    # Parameter sweep for the big-picture chart
    print(f"\n  Running parameter sweep...")
    sweep = parameter_sweep(inputs, n_iterations=500)
    print("  Done.")

    # Generate PDF report
    print("\n  Generating PDF report...")
    verdict_text = generate_verdict_text(d)
    pdf_path = report.generate_pdf(
        inputs, results, be, sweep, d, verdict_text,
        "student_loan_report.pdf",
    )
    print(f"  Saved to {pdf_path}\n")

    _print_section_6(pdf_path)


if __name__ == "__main__":
    run_cli()
