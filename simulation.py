"""
Monte Carlo simulation engine for Student Loan Plan 2 overpay-vs-invest.

Compares two strategies over the remaining years until loan write-off:
  A) Overpay the loan, invest only after it clears
  B) Accept minimum repayments, invest the overpayment from day one

All Monte Carlo iterations are vectorised with numpy; the year loop
(<=30 steps) is a plain Python loop.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config as cfg
import tax


# ─── Data Classes ─────────────────────────────────────────────────────

@dataclass
class SimInputs:
    """User inputs for the simulation."""

    loan_balance: float          # current loan balance (min £25k)
    salary: float                # current annual gross salary
    salary_growth_mean: float    # expected annual salary growth rate
    age: int                     # current age
    years_since_first_repayment: int  # years since first Plan 2 repayment
    region: str                  # 'england' or 'scotland'
    monthly_overpayment: float   # extra monthly payment towards loan
    isa: bool                    # True = ISA wrapper, False = GIA
    inv_return_mean: float       # expected annual investment return
    inv_return_std: float        # std dev of investment return
    rpi_mean: float              # expected RPI inflation rate
    rpi_std: float               # std dev of RPI inflation
    n_iterations: int = 10_000   # number of Monte Carlo iterations
    seed: int = 42               # random seed for reproducibility

    def __post_init__(self) -> None:
        if self.loan_balance < 25_000:
            raise ValueError("Loan balance must be at least £25,000")
        if self.region not in ("england", "scotland"):
            raise ValueError("Region must be 'england' or 'scotland'")
        if self.years_since_first_repayment < 0 or self.years_since_first_repayment >= 30:
            raise ValueError("Years since first repayment must be 0-29")

    @property
    def remaining_years(self) -> int:
        return cfg.SL_WRITE_OFF_YEARS - self.years_since_first_repayment


@dataclass
class SimResults:
    """Results from the Monte Carlo simulation."""

    # ── Per-year arrays: shape (remaining_years+1, n_iterations) ──
    # Index 0 = starting state, index y = state after year y

    # Scenario A — overpay
    a_loan_balance: np.ndarray = field(repr=False)
    a_investment_pot: np.ndarray = field(repr=False)
    a_net_worth: np.ndarray = field(repr=False)
    a_total_repaid: np.ndarray = field(repr=False)
    a_monthly_to_investments: np.ndarray = field(repr=False)

    # Scenario B — invest
    b_loan_balance: np.ndarray = field(repr=False)
    b_investment_pot: np.ndarray = field(repr=False)
    b_net_worth: np.ndarray = field(repr=False)
    b_total_repaid: np.ndarray = field(repr=False)
    b_monthly_to_investments: np.ndarray = field(repr=False)

    # ── Per-iteration scalars: shape (n_iterations,) ──
    a_year_loan_cleared: np.ndarray = field(repr=False)   # year loan hit 0 (NaN if never)
    a_loan_cleared: np.ndarray = field(repr=False)        # bool: cleared before write-off
    b_amount_written_off: np.ndarray = field(repr=False)  # balance forgiven at write-off

    # ── Shared stochastic paths: shape (remaining_years, n_iterations) ──
    salaries: np.ndarray = field(repr=False)
    rpi_rates: np.ndarray = field(repr=False)
    inv_returns: np.ndarray = field(repr=False)

    # ── Time axis ──
    years: np.ndarray = field(repr=False)   # 0 .. remaining_years
    ages: np.ndarray = field(repr=False)    # age .. age + remaining_years


# ─── Stochastic Path Generation ──────────────────────────────────────

def _generate_paths(inputs: SimInputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate correlated stochastic paths for all iterations.

    Returns
    -------
    inv_returns : (remaining_years, n_iterations)
    rpi_rates   : (remaining_years, n_iterations)
    salary_growth : (remaining_years, n_iterations)
    """
    rng = np.random.default_rng(inputs.seed)
    n = inputs.n_iterations
    T = inputs.remaining_years

    # --- Correlated normals for investment returns and RPI ---
    # Correlation ~0.3 between investment returns and RPI
    rho = 0.3
    # Generate independent normals then correlate
    z1 = rng.standard_normal((T, n))  # for investment returns
    z2 = rng.standard_normal((T, n))  # independent
    z2_corr = rho * z1 + np.sqrt(1 - rho**2) * z2  # correlated with z1

    # --- Investment returns with AR(1) mean reversion ---
    phi = 0.15
    inv_returns = np.empty((T, n))
    raw = inputs.inv_return_mean + inputs.inv_return_std * z1[0]
    inv_returns[0] = raw
    for t in range(1, T):
        raw = inputs.inv_return_mean + phi * (inv_returns[t - 1] - inputs.inv_return_mean) \
              + inputs.inv_return_std * z1[t]
        inv_returns[t] = raw
    inv_returns = np.clip(inv_returns, -0.50, 0.60)

    # --- RPI inflation ---
    rpi_rates = inputs.rpi_mean + inputs.rpi_std * z2_corr
    rpi_rates = np.clip(rpi_rates, 0.0, 0.12)

    # --- Salary growth ---
    salary_std = 0.02
    salary_growth = inputs.salary_growth_mean + salary_std * rng.standard_normal((T, n))
    # 3% chance of redundancy (0% growth that year)
    redundancy = rng.random((T, n)) < 0.03
    salary_growth[redundancy] = 0.0
    salary_growth = np.clip(salary_growth, -0.05, 0.15)

    return inv_returns, rpi_rates, salary_growth


# ─── GIA Tax Helpers ─────────────────────────────────────────────────

def _annual_gia_dividend_tax(
    pot: np.ndarray,
    annual_return: np.ndarray,
    gross_salary: np.ndarray,
    sim_year: int,
) -> np.ndarray:
    """Tax on dividends (30% of return) for GIA holdings."""
    total_return = pot * annual_return
    dividends = np.maximum(total_return, 0.0) * cfg.PORTFOLIO_DIVIDEND_SHARE
    taxable_income = gross_salary
    return tax.investment_tax(
        gains=np.zeros_like(dividends),
        dividends=dividends,
        taxable_income=taxable_income,
        in_isa=False,
        sim_year=sim_year,
    )


def _final_gia_cgt(
    pot: np.ndarray,
    total_contributions: np.ndarray,
    deferred_exempt_used: np.ndarray,
    gross_salary: np.ndarray,
    sim_year: int,
) -> np.ndarray:
    """CGT on deferred capital gains at the end of the simulation.

    The 70% capital growth share accumulates over all years. Each year
    we assume the £3,000 annual exempt amount is used. At the end,
    the total gain minus exemptions already used is taxed.
    """
    total_gain = np.maximum(pot - total_contributions, 0.0)
    capital_gain = total_gain * cfg.PORTFOLIO_CAPITAL_GROWTH_SHARE
    # Subtract the cumulative annual exemptions already consumed
    taxable_gain = np.maximum(capital_gain - deferred_exempt_used, 0.0)
    return tax.investment_tax(
        gains=taxable_gain,
        dividends=np.zeros_like(taxable_gain),
        taxable_income=gross_salary,
        in_isa=False,
        sim_year=sim_year,
    )


# ─── Core Simulation ─────────────────────────────────────────────────

def run_simulation(inputs: SimInputs) -> SimResults:
    """Run the Monte Carlo simulation comparing overpay vs invest."""

    T = inputs.remaining_years
    n = inputs.n_iterations
    base_year = cfg.BASE_TAX_YEAR

    # Generate shared stochastic paths
    inv_returns, rpi_rates, salary_growth = _generate_paths(inputs)

    # Build salary paths: shape (T, n)
    salaries = np.empty((T, n))
    salaries[0] = inputs.salary * (1 + salary_growth[0])
    for t in range(1, T):
        salaries[t] = salaries[t - 1] * (1 + salary_growth[t])

    # ── Initialise output arrays ──────────────────────────────────
    # Shape: (T+1, n) — index 0 is starting state
    shape = (T + 1, n)

    a_loan = np.zeros(shape)
    a_pot = np.zeros(shape)
    a_repaid = np.zeros(shape)
    a_monthly_inv = np.zeros(shape)

    b_loan = np.zeros(shape)
    b_pot = np.zeros(shape)
    b_repaid = np.zeros(shape)
    b_monthly_inv = np.zeros(shape)

    # Starting state
    a_loan[0] = inputs.loan_balance
    b_loan[0] = inputs.loan_balance

    # Track whether Scenario A loan has cleared (per iteration)
    a_cleared = np.zeros(n, dtype=bool)
    a_year_cleared = np.full(n, np.nan)

    # GIA tracking: total contributions and cumulative CGT exempt used
    a_contributions = np.zeros(n)
    b_contributions = np.zeros(n)
    a_cgt_exempt_used = np.zeros(n)
    b_cgt_exempt_used = np.zeros(n)

    annual_overpayment = inputs.monthly_overpayment * 12

    # ── Year loop ─────────────────────────────────────────────────
    for y in range(T):
        sim_year = base_year + y
        sal = salaries[y]  # (n,)
        rpi = rpi_rates[y]  # (n,)
        ret = inv_returns[y]  # (n,)

        # Mandatory SL repayment (same salary, same deduction for both)
        # Use per-iteration RPI for threshold calculation — but the function
        # expects a scalar RPI. We use the mean RPI for threshold (threshold
        # is a policy parameter, not stochastic per-person). This is a
        # simplification; in reality the threshold is set once per year.
        rpi_scalar = float(np.mean(rpi))
        mandatory_repay = tax.student_loan_repayment(sal, sim_year, rpi_scalar)

        # SL interest rate (vectorised over salary, scalar RPI)
        sl_interest = tax.student_loan_interest_rate(sal, rpi_scalar, sim_year)

        # ──── Scenario A: Overpay ────────────────────────────────
        # Loans that are still active
        a_active = ~a_cleared  # bool (n,)

        # Interest accrues on remaining balance
        a_bal = a_loan[y].copy()
        a_bal_with_interest = a_bal * (1 + sl_interest)

        # Apply mandatory repayment + overpayment (only while loan active)
        a_total_payment = np.where(a_active, mandatory_repay + annual_overpayment, 0.0)
        # Don't overpay beyond balance
        a_total_payment = np.minimum(a_total_payment, np.maximum(a_bal_with_interest, 0.0))

        a_bal_after = np.where(a_active, a_bal_with_interest - a_total_payment, 0.0)
        a_bal_after = np.maximum(a_bal_after, 0.0)

        # Detect loans that just cleared this year
        just_cleared = a_active & (a_bal_after <= 0.01)
        a_cleared = a_cleared | just_cleared
        a_year_cleared = np.where(just_cleared & np.isnan(a_year_cleared), y + 1, a_year_cleared)
        a_bal_after[just_cleared] = 0.0

        a_loan[y + 1] = a_bal_after
        a_repaid[y + 1] = a_repaid[y] + a_total_payment

        # Investment: only once loan is cleared
        # When cleared, both mandatory repayment AND overpayment go to investments
        a_inv_amount = np.where(a_cleared, mandatory_repay + annual_overpayment, 0.0)
        a_monthly_inv[y + 1] = np.where(a_cleared, (mandatory_repay + annual_overpayment) / 12, 0.0)

        # Grow existing pot + add new contributions
        # Contributions arrive mid-year on average, so get ~half the return
        a_prev_pot = a_pot[y]
        if inputs.isa:
            a_pot_grown = a_prev_pot * (1 + ret) + a_inv_amount * (1 + ret * 0.5)
        else:
            # GIA: deduct annual dividend tax from pot growth
            div_tax = _annual_gia_dividend_tax(a_prev_pot, np.maximum(ret, 0.0), sal, sim_year)
            a_pot_grown = a_prev_pot * (1 + ret) - div_tax + a_inv_amount * (1 + ret * 0.5)
            # Track contributions and CGT exemptions
            a_contributions += a_inv_amount
            yearly_growth = np.maximum(a_prev_pot * ret, 0.0) * cfg.PORTFOLIO_CAPITAL_GROWTH_SHARE
            a_cgt_exempt_used += np.minimum(yearly_growth, cfg.CGT_ANNUAL_EXEMPT)

        a_pot[y + 1] = np.maximum(a_pot_grown, 0.0)

        # ──── Scenario B: Invest from day one ────────────────────
        b_active = b_loan[y] > 0.01  # bool (n,)

        # Interest on loan
        b_bal = b_loan[y].copy()
        b_bal_with_interest = b_bal * (1 + sl_interest)

        # Only mandatory repayment
        b_payment = np.where(b_active, np.minimum(mandatory_repay, np.maximum(b_bal_with_interest, 0.0)), 0.0)
        b_bal_after = np.where(b_active, b_bal_with_interest - b_payment, 0.0)
        b_bal_after = np.maximum(b_bal_after, 0.0)
        b_loan[y + 1] = b_bal_after
        b_repaid[y + 1] = b_repaid[y] + b_payment

        # Invest the overpayment amount (always, from year 1)
        # After loan clears in Scenario B, mandatory repayment also goes to investments
        b_loan_active = b_loan[y + 1] > 0.01
        b_inv_amount = np.where(b_loan_active, annual_overpayment, annual_overpayment + mandatory_repay)
        b_monthly_inv[y + 1] = b_inv_amount / 12

        b_prev_pot = b_pot[y]
        if inputs.isa:
            b_pot_grown = b_prev_pot * (1 + ret) + b_inv_amount * (1 + ret * 0.5)
        else:
            div_tax = _annual_gia_dividend_tax(b_prev_pot, np.maximum(ret, 0.0), sal, sim_year)
            b_pot_grown = b_prev_pot * (1 + ret) - div_tax + b_inv_amount * (1 + ret * 0.5)
            b_contributions += b_inv_amount
            yearly_growth = np.maximum(b_prev_pot * ret, 0.0) * cfg.PORTFOLIO_CAPITAL_GROWTH_SHARE
            b_cgt_exempt_used += np.minimum(yearly_growth, cfg.CGT_ANNUAL_EXEMPT)

        b_pot[y + 1] = np.maximum(b_pot_grown, 0.0)

    # ── Final GIA CGT adjustment ──────────────────────────────────
    if not inputs.isa:
        final_year = base_year + T - 1
        final_sal = salaries[-1]

        a_cgt = _final_gia_cgt(a_pot[T], a_contributions, a_cgt_exempt_used, final_sal, final_year)
        a_pot[T] = np.maximum(a_pot[T] - a_cgt, 0.0)

        b_cgt = _final_gia_cgt(b_pot[T], b_contributions, b_cgt_exempt_used, final_sal, final_year)
        b_pot[T] = np.maximum(b_pot[T] - b_cgt, 0.0)

    # ── Net worth = investment pot (loan written off at end) ──────
    a_net_worth = a_pot.copy()
    b_net_worth = b_pot.copy()

    # Amount written off in Scenario B = remaining balance at write-off
    b_written_off = b_loan[T]

    # Time arrays
    years = np.arange(T + 1)
    ages = inputs.age + years

    return SimResults(
        a_loan_balance=a_loan,
        a_investment_pot=a_pot,
        a_net_worth=a_net_worth,
        a_total_repaid=a_repaid,
        a_monthly_to_investments=a_monthly_inv,
        a_year_loan_cleared=a_year_cleared,
        a_loan_cleared=a_cleared,
        b_loan_balance=b_loan,
        b_investment_pot=b_pot,
        b_net_worth=b_net_worth,
        b_total_repaid=b_repaid,
        b_monthly_to_investments=b_monthly_inv,
        b_amount_written_off=b_written_off,
        salaries=salaries,
        rpi_rates=rpi_rates,
        inv_returns=inv_returns,
        years=years,
        ages=ages,
    )


# ─── Breakeven Row ────────────────────────────────────────────────────

@dataclass
class BreakevenRow:
    """One row of the personal breakeven table."""

    monthly_overpayment: float
    pct_loan_clears: float
    median_year_cleared: Optional[float]   # None if < 50% clear
    median_age_cleared: Optional[float]    # None if < 50% clear
    median_nw_overpay: float
    median_nw_invest: float
    winner: str                            # 'overpay' or 'invest'
    advantage: float                       # positive = winner's margin


@dataclass
class BreakevenResult:
    """Output of the personal breakeven table."""

    rows: list[BreakevenRow]
    breakeven_amount: Optional[float]      # None if overpay never wins


@dataclass
class SweepResult:
    """Output of the full parameter sweep."""

    loan_balances: np.ndarray              # (n_loans,)
    salaries: np.ndarray                   # (n_salaries,)
    overpayments: np.ndarray               # (n_overpayments,)
    advantage: np.ndarray                  # (n_loans, n_salaries, n_overpayments)
    # positive = investing wins


# ─── Function 1: Personal Breakeven Table ─────────────────────────────

BREAKEVEN_OVERPAYMENTS = [0, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]


def breakeven_table(inputs: SimInputs) -> BreakevenResult:
    """Test overpayment levels and find the breakeven point.

    Runs 1,000-iteration simulations for each overpayment level using
    the user's other parameters, then identifies the lowest monthly
    overpayment where the overpay strategy wins > 50% of simulations.

    Parameters
    ----------
    inputs : SimInputs
        Base user inputs. ``monthly_overpayment`` and ``n_iterations``
        will be overridden per level.

    Returns
    -------
    BreakevenResult
        Table rows and breakeven amount (None if overpay never wins).
    """
    rows: list[BreakevenRow] = []
    breakeven: Optional[float] = None

    for overpay in BREAKEVEN_OVERPAYMENTS:
        # Clone inputs with this overpayment level and reduced iterations
        inp = SimInputs(
            loan_balance=inputs.loan_balance,
            salary=inputs.salary,
            salary_growth_mean=inputs.salary_growth_mean,
            age=inputs.age,
            years_since_first_repayment=inputs.years_since_first_repayment,
            region=inputs.region,
            monthly_overpayment=overpay,
            isa=inputs.isa,
            inv_return_mean=inputs.inv_return_mean,
            inv_return_std=inputs.inv_return_std,
            rpi_mean=inputs.rpi_mean,
            rpi_std=inputs.rpi_std,
            n_iterations=1_000,
            seed=inputs.seed,
        )
        res = run_simulation(inp)
        T = inp.remaining_years

        # Loan clearance stats
        pct_cleared = float(res.a_loan_cleared.mean() * 100)
        cleared_years = res.a_year_loan_cleared[~np.isnan(res.a_year_loan_cleared)]
        if pct_cleared >= 50.0 and len(cleared_years) > 0:
            med_year = float(np.median(cleared_years))
            med_age = float(inputs.age + med_year)
        else:
            med_year = None
            med_age = None

        # Net worth comparison
        med_nw_a = float(np.median(res.a_net_worth[T]))
        med_nw_b = float(np.median(res.b_net_worth[T]))

        overpay_wins_pct = float((res.a_net_worth[T] > res.b_net_worth[T]).mean() * 100)

        if overpay_wins_pct > 50:
            winner = "overpay"
            advantage = med_nw_a - med_nw_b
        else:
            winner = "invest"
            advantage = med_nw_b - med_nw_a

        rows.append(BreakevenRow(
            monthly_overpayment=overpay,
            pct_loan_clears=pct_cleared,
            median_year_cleared=med_year,
            median_age_cleared=med_age,
            median_nw_overpay=med_nw_a,
            median_nw_invest=med_nw_b,
            winner=winner,
            advantage=advantage,
        ))

        if breakeven is None and overpay_wins_pct > 50:
            breakeven = float(overpay)

    return BreakevenResult(rows=rows, breakeven_amount=breakeven)


# ─── Function 2: Full Parameter Sweep ─────────────────────────────────

SWEEP_LOANS = [25_000, 35_000, 45_000, 55_000, 65_000]
SWEEP_SALARIES = [30_000, 40_000, 50_000, 60_000, 80_000, 100_000]
SWEEP_OVERPAYMENTS = [0, 100, 200, 300, 500, 750, 1_000]


def parameter_sweep(
    base_inputs: SimInputs,
    n_iterations: int = 500,
) -> SweepResult:
    """Sweep loan × salary × overpayment grid and compute median advantage.

    For each combination, runs a reduced-iteration simulation and records
    the median advantage of investing over overpaying (positive = invest wins).

    Parameters
    ----------
    base_inputs : SimInputs
        Template for non-swept parameters (age, region, growth rates, etc.).
    n_iterations : int
        Iterations per combo (default 500 for speed).

    Returns
    -------
    SweepResult
        3D array of advantages plus axis values.
    """
    loans = np.array(SWEEP_LOANS, dtype=float)
    sals = np.array(SWEEP_SALARIES, dtype=float)
    overpays = np.array(SWEEP_OVERPAYMENTS, dtype=float)

    n_combos = len(loans) * len(sals) * len(overpays)
    advantage = np.empty((len(loans), len(sals), len(overpays)))

    done = 0
    t0 = time.time()

    for i, loan in enumerate(loans):
        for j, sal in enumerate(sals):
            for k, overpay in enumerate(overpays):
                inp = SimInputs(
                    loan_balance=loan,
                    salary=sal,
                    salary_growth_mean=base_inputs.salary_growth_mean,
                    age=base_inputs.age,
                    years_since_first_repayment=base_inputs.years_since_first_repayment,
                    region=base_inputs.region,
                    monthly_overpayment=overpay,
                    isa=base_inputs.isa,
                    inv_return_mean=base_inputs.inv_return_mean,
                    inv_return_std=base_inputs.inv_return_std,
                    rpi_mean=base_inputs.rpi_mean,
                    rpi_std=base_inputs.rpi_std,
                    n_iterations=n_iterations,
                    seed=base_inputs.seed,
                )
                res = run_simulation(inp)
                T = inp.remaining_years

                # Median advantage of investing (positive = invest wins)
                adv = np.median(res.b_net_worth[T] - res.a_net_worth[T])
                advantage[i, j, k] = adv

                done += 1
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_combos - done) / rate if rate > 0 else 0
                bar_len = 30
                filled = int(bar_len * done / n_combos)
                bar = "#" * filled + "-" * (bar_len - filled)
                sys.stdout.write(
                    f"\r  [{bar}] {done}/{n_combos} "
                    f"({done/n_combos*100:.0f}%) "
                    f"ETA {eta:.0f}s"
                )
                sys.stdout.flush()

    sys.stdout.write("\n")
    elapsed = time.time() - t0
    print(f"  Sweep completed in {elapsed:.1f}s ({n_combos} combos, "
          f"{elapsed/n_combos:.2f}s each)")

    return SweepResult(
        loan_balances=loans,
        salaries=sals,
        overpayments=overpays,
        advantage=advantage,
    )


# ─── Tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Monte Carlo Simulation — Smoke Test")
    print("=" * 60)

    inputs = SimInputs(
        loan_balance=45_000,
        salary=35_000,
        salary_growth_mean=0.04,
        age=25,
        years_since_first_repayment=1,
        region="england",
        monthly_overpayment=200,
        isa=True,
        inv_return_mean=0.07,
        inv_return_std=0.15,
        rpi_mean=0.032,
        rpi_std=0.015,
        n_iterations=1_000,
        seed=42,
    )

    results = run_simulation(inputs)

    T = inputs.remaining_years
    print(f"\nRemaining years until write-off: {T}")
    print(f"Ages: {results.ages[0]} -> {results.ages[-1]}")

    # ── Scenario A ──
    pct_cleared = results.a_loan_cleared.mean() * 100
    cleared_iters = results.a_year_loan_cleared[~np.isnan(results.a_year_loan_cleared)]
    median_year_cleared = np.median(cleared_iters) if len(cleared_iters) > 0 else float("nan")
    median_age_cleared = inputs.age + median_year_cleared if not np.isnan(median_year_cleared) else float("nan")
    a_final_nw = np.median(results.a_net_worth[T])

    print(f"\n--- Scenario A: Overpay ---")
    print(f"  Loan clears before write-off: {pct_cleared:.1f}% of simulations")
    print(f"  Median year cleared:          {median_year_cleared:.0f}")
    print(f"  Median age at clearance:      {median_age_cleared:.0f}")
    print(f"  Median final net worth:       £{a_final_nw:,.0f}")

    # ── Scenario B ──
    b_final_nw = np.median(results.b_net_worth[T])
    b_written_off = np.median(results.b_amount_written_off)

    print(f"\n--- Scenario B: Invest ---")
    print(f"  Median final net worth:       £{b_final_nw:,.0f}")
    print(f"  Median amount written off:    £{b_written_off:,.0f}")

    # ── Comparison ──
    invest_wins = results.b_net_worth[T] > results.a_net_worth[T]
    pct_invest_wins = invest_wins.mean() * 100
    advantage = results.b_net_worth[T] - results.a_net_worth[T]
    median_advantage = np.median(advantage)

    print(f"\n--- Comparison ---")
    print(f"  Investing wins: {pct_invest_wins:.1f}% of simulations")
    print(f"  Median advantage (B - A): £{median_advantage:,.0f}")

    # ── Sanity checks ──
    print(f"\n--- Sanity Checks ---")
    checks_passed = 0
    checks_total = 0

    def verify(name: str, condition: bool) -> None:
        global checks_passed, checks_total
        checks_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            checks_passed += 1
        print(f"  [{status}] {name}")

    verify("A: investment pot zero at year 1", np.all(results.a_investment_pot[1] == 0.0))
    verify("B: investment pot > 0 at year 1", np.all(results.b_investment_pot[1] > 0))
    verify("A: no NaN in net worth", not np.any(np.isnan(results.a_net_worth)))
    verify("A: no Inf in net worth", not np.any(np.isinf(results.a_net_worth)))
    verify("B: no NaN in net worth", not np.any(np.isnan(results.b_net_worth)))
    verify("B: no Inf in net worth", not np.any(np.isinf(results.b_net_worth)))
    verify("A: loan balance >= 0", np.all(results.a_loan_balance >= 0))
    verify("B: loan balance >= 0", np.all(results.b_loan_balance >= 0))
    verify("A: median NW < £500k", a_final_nw < 500_000)
    verify("B: median NW < £500k", b_final_nw < 500_000)
    verify("A: median NW >= 0", a_final_nw >= 0)
    verify("B: median NW >= 0", b_final_nw >= 0)
    median_final_salary = np.median(results.salaries[-1])
    verify(f"Final median salary reasonable (£{median_final_salary:,.0f})",
           20_000 < median_final_salary < 200_000)

    print(f"\n  Simulation checks: {checks_passed}/{checks_total} passed")

    # ══════════════════════════════════════════════════════════════
    # Function 1: Personal Breakeven Table
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Personal Breakeven Table")
    print("=" * 60)

    be = breakeven_table(inputs)

    # Table header
    header = (
        f"{'Overpay':>8}  {'Clears%':>7}  {'Yr':>4}  {'Age':>4}  "
        f"{'NW(Overpay)':>12}  {'NW(Invest)':>12}  {'Winner':>7}  {'By':>10}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for r in be.rows:
        yr_str = f"{r.median_year_cleared:.0f}" if r.median_year_cleared is not None else "  - "
        age_str = f"{r.median_age_cleared:.0f}" if r.median_age_cleared is not None else "  - "
        print(
            f"£{r.monthly_overpayment:>6,.0f}  {r.pct_loan_clears:>6.1f}%  "
            f"{yr_str:>4}  {age_str:>4}  "
            f"£{r.median_nw_overpay:>10,.0f}  £{r.median_nw_invest:>10,.0f}  "
            f"{r.winner:>7}  £{r.advantage:>8,.0f}"
        )

    if be.breakeven_amount is not None:
        print(f"\n  Breakeven overpayment: £{be.breakeven_amount:,.0f}/month")
    else:
        print(f"\n  Breakeven: investing wins at all levels up to £1,000/month")

    # Sanity checks for breakeven
    print(f"\n--- Breakeven Sanity Checks ---")
    be_checks_passed = 0
    be_checks_total = 0

    def verify_be(name: str, condition: bool) -> None:
        global be_checks_passed, be_checks_total
        be_checks_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            be_checks_passed += 1
        print(f"  [{status}] {name}")

    verify_be("Got 11 rows", len(be.rows) == 11)
    verify_be("No NaN in net worth values",
              all(not np.isnan(r.median_nw_overpay) and not np.isnan(r.median_nw_invest) for r in be.rows))
    verify_be("£0 overpay: both NW are 0 (nothing invested either way)",
              be.rows[0].median_nw_overpay == 0.0 and be.rows[0].median_nw_invest == 0.0)
    verify_be("Higher overpay -> higher overpay NW (monotonic from £150+)",
              all(be.rows[i+1].median_nw_overpay >= be.rows[i].median_nw_overpay
                  for i in range(3, len(be.rows) - 1)))
    verify_be("Invest NW scales with overpay amount",
              be.rows[-1].median_nw_invest > be.rows[1].median_nw_invest)

    print(f"\n  Breakeven checks: {be_checks_passed}/{be_checks_total} passed")

    # ══════════════════════════════════════════════════════════════
    # Function 2: Parameter Sweep
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Parameter Sweep (210 combos × 500 iterations)")
    print("=" * 60)
    print()

    sweep = parameter_sweep(inputs, n_iterations=500)

    # Summary: for each loan balance, which combos favour overpaying
    print(f"\n--- Combos where overpaying wins (advantage < 0) ---\n")

    overpay_wins_count = 0
    total_combos = 0

    for i, loan in enumerate(sweep.loan_balances):
        favours = []
        for j, sal in enumerate(sweep.salaries):
            for k, overpay in enumerate(sweep.overpayments):
                total_combos += 1
                adv = sweep.advantage[i, j, k]
                if adv < 0:
                    overpay_wins_count += 1
                    favours.append(
                        f"    sal=£{sal:,.0f}, overpay=£{overpay:,.0f}/mo "
                        f"-> overpay wins by £{-adv:,.0f}"
                    )
        print(f"  Loan £{loan:,.0f}: ", end="")
        if favours:
            print(f"{len(favours)} combo(s) favour overpaying:")
            for line in favours:
                print(line)
        else:
            print("investing wins at ALL salary/overpayment levels")

    print(f"\n  Total: {overpay_wins_count}/{total_combos} combos favour overpaying")

    # Sweep sanity checks
    print(f"\n--- Sweep Sanity Checks ---")
    sw_checks_passed = 0
    sw_checks_total = 0

    def verify_sw(name: str, condition: bool) -> None:
        global sw_checks_passed, sw_checks_total
        sw_checks_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            sw_checks_passed += 1
        print(f"  [{status}] {name}")

    verify_sw("Shape correct (5×6×7)", sweep.advantage.shape == (5, 6, 7))
    verify_sw("No NaN values", not np.any(np.isnan(sweep.advantage)))
    verify_sw("No Inf values", not np.any(np.isinf(sweep.advantage)))
    verify_sw("£0 overpay always 0 advantage (both scenarios identical)",
              np.all(np.abs(sweep.advantage[:, :, 0]) < 1.0))
    verify_sw("Values in reasonable range (< £500k)",
              np.all(np.abs(sweep.advantage) < 500_000))

    print(f"\n  Sweep checks: {sw_checks_passed}/{sw_checks_total} passed")
    print(f"\n{'='*60}")
    total_all = checks_total + be_checks_total + sw_checks_total
    passed_all = checks_passed + be_checks_passed + sw_checks_passed
    print(f"ALL CHECKS: {passed_all}/{total_all} passed")
    print("=" * 60)
