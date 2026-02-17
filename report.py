"""
PDF report generation and reusable chart rendering for the
Student Loan Plan 2 overpay-vs-invest simulator.

Provides:
  - Six-page PDF report (generate_pdf)
  - Base64-encoded chart images for web embedding (get_web_charts)
  - Individual page renderers reusable by both CLI and web
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import numpy as np

import config as cfg
import tax
from simulation import (
    BreakevenResult,
    SimInputs,
    SimResults,
    SweepResult,
)

# ═══════════════════════════════════════════════════════════════════
# Style constants
# ═══════════════════════════════════════════════════════════════════

BG = "#0a101f"
CARD = "#131b2e"
TEXT = "#f1f5f9"
TEXT2 = "#cbd5e1"
INDIGO = "#818cf8"
EMERALD = "#34d399"
AMBER = "#fbbf24"
RED = "#f87171"
SLATE = "#94a3b8"
BORDER = "#1e293b"
INDIGO_DEEP = "#6366f1"
EMERALD_DEEP = "#10b981"

A4W, A4H = 8.27, 11.69
WEB_W, WEB_H = 10, 7


# ═══════════════════════════════════════════════════════════════════
# Axis formatters
# ═══════════════════════════════════════════════════════════════════

def _gbp_fmt(x, _):
    if abs(x) >= 1e6:
        return f"\u00a3{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"\u00a3{x / 1e3:.0f}k"
    return f"\u00a3{x:.0f}"


def _pct_fmt(x, _):
    return f"{x:.0f}%"


GBP_FMT = FuncFormatter(_gbp_fmt)
PCT_FMT = FuncFormatter(_pct_fmt)


# ═══════════════════════════════════════════════════════════════════
# Style helpers
# ═══════════════════════════════════════════════════════════════════

def _style(fig, *axes):
    """Apply dark theme to figure and all axes."""
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(True, alpha=0.15, color=SLATE)


def _add_year_axis(ax, ages, years):
    """Add simulation-year labels on secondary top x-axis."""
    T = len(ages) - 1
    sp = max(1, T // 6)
    idxs = list(range(0, T + 1, sp))
    if idxs[-1] != T:
        idxs.append(T)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([ages[i] for i in idxs])
    ax2.set_xticklabels([f"Yr {years[i]}" for i in idxs], fontsize=7)
    ax2.tick_params(colors=SLATE, labelsize=7, length=3)
    for spine in ax2.spines.values():
        spine.set_color(BORDER)
    return ax2


def _fan(ax, x, data_2d, color, label):
    """Draw fan chart: median + 25-75 band + 5-95 band. Returns median."""
    med = np.median(data_2d, axis=1)
    p25, p75 = np.percentile(data_2d, [25, 75], axis=1)
    p5, p95 = np.percentile(data_2d, [5, 95], axis=1)
    ax.fill_between(x, p5, p95, color=color, alpha=0.10)
    ax.fill_between(x, p25, p75, color=color, alpha=0.28)
    ax.plot(x, med, color=color, linewidth=2.2, label=label, solid_capstyle="round")
    return med


def _legend(ax, loc="upper left"):
    ax.legend(loc=loc, fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)


# ═══════════════════════════════════════════════════════════════════
# Page 1 — Summary (text only, dark background)
# ═══════════════════════════════════════════════════════════════════

def _page1_summary(inputs: SimInputs, d: Dict, verdict_text: str,
                   be: BreakevenResult) -> plt.Figure:
    fig = plt.figure(figsize=(A4W, A4H))
    fig.patch.set_facecolor(BG)

    # Title
    fig.text(0.50, 0.93, "Student Loan Plan 2: Overpay vs Invest",
             ha="center", fontsize=18, color=TEXT, fontweight="bold")
    fig.text(0.50, 0.91, "Monte Carlo Simulation Report",
             ha="center", fontsize=11, color=TEXT2)

    y = 0.86
    # Parameters
    fig.text(0.08, y, "Your Parameters",
             fontsize=13, color=TEXT, fontweight="bold")
    y -= 0.028
    params = [
        f"Age: {inputs.age}  |  Salary: \u00a3{inputs.salary:,.0f}  |  "
        f"Loan: \u00a3{inputs.loan_balance:,.0f}  |  Region: {inputs.region.title()}",
        f"Overpayment: \u00a3{inputs.monthly_overpayment:,.0f}/mo  |  "
        f"Wrapper: {'ISA' if inputs.isa else 'GIA'}  |  "
        f"Expected return: {inputs.inv_return_mean * 100:.1f}%  |  "
        f"Iterations: {inputs.n_iterations:,}",
    ]
    for p in params:
        fig.text(0.10, y, p, fontsize=9, color=TEXT2)
        y -= 0.024

    # Option A
    y -= 0.025
    fig.text(0.08, y,
             f"Option A \u2014 Overpay \u00a3{inputs.monthly_overpayment:,.0f}/mo",
             fontsize=13, color=INDIGO, fontweight="bold")
    y -= 0.028
    a_lines = [
        f"Net worth at write-off (median): \u00a3{d['a_nw']:,.0f}",
        f"Loan clears: {d['a_pct_cleared']:.0f}% of simulations",
        f"Total paid to SLC: \u00a3{d['a_paid']:,.0f}",
    ]
    if d["a_med_age"] is not None:
        a_lines.insert(1, f"Clears at age (median): {d['a_med_age']:.0f}")
    for line in a_lines:
        fig.text(0.10, y, line, fontsize=9.5, color=TEXT2)
        y -= 0.024

    # Freed cashflow reinvestment
    if d["a_years_investing"] > 0:
        y -= 0.008
        fig.text(0.10, y, "Freed Cashflow Reinvested After Loan Clears:",
                 fontsize=9, color=INDIGO, fontweight="bold")
        y -= 0.022
        freed_lines = [
            f"\u00a3{d['a_freed_overpayment']:,.0f}/mo overpayment + "
            f"\u00a3{d['a_freed_mandatory']:,.0f}/mo mandatory = "
            f"\u00a3{d['a_monthly_invest']:,.0f}/mo total reinvested",
            f"Duration: {d['a_years_investing']:.0f} years  |  "
            f"Total contributions: \u00a3{d['a_total_reinvested']:,.0f}",
        ]
        for line in freed_lines:
            fig.text(0.12, y, line, fontsize=8.5, color=TEXT2)
            y -= 0.021

    # Option B
    y -= 0.025
    fig.text(0.08, y,
             f"Option B \u2014 Invest \u00a3{inputs.monthly_overpayment:,.0f}/mo Instead",
             fontsize=13, color=EMERALD, fontweight="bold")
    y -= 0.028
    b_lines = [
        f"Net worth at write-off (median): \u00a3{d['b_nw']:,.0f}",
        f"Amount forgiven: \u00a3{d['b_forgiven']:,.0f}",
        f"Total paid to SLC: \u00a3{d['b_paid']:,.0f}",
    ]
    for line in b_lines:
        fig.text(0.10, y, line, fontsize=9.5, color=TEXT2)
        y -= 0.024

    # Verdict
    y -= 0.03
    winner_color = EMERALD if d["winner"] == "invest" else INDIGO
    winner_label = "INVESTING WINS" if d["winner"] == "invest" else "OVERPAYING WINS"
    fig.text(0.08, y, "The Verdict",
             fontsize=13, color=TEXT, fontweight="bold")
    y -= 0.03
    fig.text(0.10, y,
             f"{winner_label} \u2014 by \u00a3{d['adv_abs']:,.0f} ({d['adv_pct']:.1f}%)",
             fontsize=12, color=winner_color, fontweight="bold")
    y -= 0.03

    # Word-wrap verdict text
    words = verdict_text.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 <= 85:
            line = f"{line} {word}" if line else word
        else:
            fig.text(0.10, y, line, fontsize=9, color=TEXT2)
            y -= 0.022
            line = word
    if line:
        fig.text(0.10, y, line, fontsize=9, color=TEXT2)
        y -= 0.022

    # Invest-favorable threshold
    y -= 0.025
    fig.text(0.08, y, "Invest-Favorable Threshold",
             fontsize=13, color=EMERALD, fontweight="bold")
    y -= 0.028
    if d.get("invest_favorable_below") is not None:
        threshold_text = (
            f"Investing is favorable below "
            f"\u00a3{d['invest_favorable_below']:,.0f}/mo. "
            f"You'd need to overpay \u00a3{d['invest_favorable_below']:,.0f}/mo+ "
            f"to beat investing."
        )
    else:
        threshold_text = (
            f"Investing is favorable at every level tested "
            f"(up to \u00a3{d['invest_favorable_max_tested']:,.0f}/mo)."
        )
    fig.text(0.10, y, threshold_text, fontsize=9.5, color=EMERALD)
    y -= 0.024

    # Breakeven
    y -= 0.02
    fig.text(0.08, y, "Breakeven Table",
             fontsize=13, color=TEXT, fontweight="bold")
    y -= 0.028
    if be.breakeven_amount is not None:
        be_text = (f"You'd need to overpay \u00a3{be.breakeven_amount:,.0f}/mo+ "
                   f"for it to beat investing.")
    else:
        be_text = ("No amount up to \u00a31,000/mo beats investing "
                   "\u2014 your loan will be written off.")
    fig.text(0.10, y, be_text, fontsize=10, color=AMBER)

    # Lump sum hypothetical
    y -= 0.04
    fig.text(0.08, y, "Hypothetical: Lump Sum Payoff",
             fontsize=13, color="#c4b5fd", fontweight="bold")
    y -= 0.028
    ls_lines = [
        f"If you had \u00a3{d['ls_lump_sum']:,.0f} in cash today:",
        f"  Pay off now \u2192 reinvest freed \u00a3{d['ls_freed_monthly']:,.0f}/mo "
        f"for {d['ls_years']} years \u2192 \u00a3{d['ls_payoff_fv']:,.0f}",
        f"  Invest lump sum \u2192 \u00a3{d['ls_lump_sum']:,.0f} grows "
        f"for {d['ls_years']} years at {d['ls_return'] * 100:.1f}% "
        f"\u2192 \u00a3{d['ls_invest_fv']:,.0f}",
    ]
    ls_winner_color = EMERALD if d["ls_winner"] == "invest" else INDIGO
    ls_winner_label = ("Investing the lump sum wins"
                       if d["ls_winner"] == "invest" else "Paying off wins")
    for line in ls_lines:
        fig.text(0.10, y, line, fontsize=9, color=TEXT2)
        y -= 0.022
    y -= 0.005
    fig.text(0.10, y,
             f"{ls_winner_label} by \u00a3{d['ls_advantage']:,.0f} "
             f"({d['ls_adv_pct']:.1f}%)",
             fontsize=10, color=ls_winner_color, fontweight="bold")

    # Disclaimer
    fig.text(0.50, 0.03,
             "This is not financial advice. Past performance does not "
             "guarantee future results.",
             ha="center", fontsize=8, color=SLATE, style="italic")
    return fig


# ═══════════════════════════════════════════════════════════════════
# Page 2 — Tax Position (stacked bars + effective rate line)
# ═══════════════════════════════════════════════════════════════════

def _page2_tax(inputs: SimInputs, results: SimResults,
               figsize=(A4W, A4H)) -> plt.Figure:
    T = inputs.remaining_years
    ages = results.ages
    years = results.years
    sal_path = np.concatenate(
        [[inputs.salary], np.median(results.salaries, axis=1)]
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                   constrained_layout=True)
    _style(fig, ax1, ax2)

    # ── Top: stacked bar of marginal rates at salary checkpoints ──
    checkpoints = [0]
    for yr in [10, 20]:
        if yr < T:
            checkpoints.append(yr)
    if T > 0 and (T - 1) not in checkpoints:
        checkpoints.append(T - 1)

    bar_labels, it_vals, ni_vals, sl_vals = [], [], [], []
    for yr in checkpoints:
        sim_year = cfg.BASE_TAX_YEAR + yr
        sal = sal_path[yr]
        mb = tax.marginal_rate_breakdown(sal, inputs.region, sim_year,
                                         rpi=inputs.rpi_mean)
        it_vals.append(mb["income_tax_pct"])
        ni_vals.append(mb["ni_pct"])
        sl_vals.append(mb["sl_pct"])
        age = inputs.age + yr
        tag = "Now" if yr == 0 else f"Yr {yr}"
        bar_labels.append(f"{tag}\n\u00a3{sal:,.0f}\n(age {age})")

    x = np.arange(len(checkpoints))
    w = 0.5
    it_arr = np.array(it_vals)
    ni_arr = np.array(ni_vals)
    sl_arr = np.array(sl_vals)

    ax1.bar(x, it_arr, w, color=INDIGO, label="Income Tax")
    ax1.bar(x, ni_arr, w, bottom=it_arr, color=SLATE,
            label="National Insurance")
    ax1.bar(x, sl_arr, w, bottom=it_arr + ni_arr, color=AMBER,
            hatch="//", edgecolor=AMBER, linewidth=0.5,
            label="Student Loan")

    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_labels, fontsize=7.5)
    ax1.set_ylim(0, 80)
    ax1.set_ylabel("Marginal Rate")
    ax1.yaxis.set_major_formatter(PCT_FMT)
    ax1.axhline(50, color=RED, linewidth=1, linestyle="--", alpha=0.5)
    ax1.text(len(checkpoints) - 0.6, 51, "50%", fontsize=7,
             color=RED, alpha=0.7)
    ax1.set_title("Marginal Rate Breakdown", fontsize=11, pad=10)
    _legend(ax1)

    # ── Bottom: effective deduction rate over time ──
    rate_with_sl = np.zeros(T + 1)
    rate_without_sl = np.zeros(T + 1)
    for yr in range(T + 1):
        sim_year = cfg.BASE_TAX_YEAR + yr
        s = np.array([sal_path[min(yr, T - 1) if yr == T else yr]])
        it_v = float(tax.income_tax(s, inputs.region, sim_year)[0])
        ni_v = float(tax.national_insurance(s, sim_year)[0])
        sl_v = float(tax.student_loan_repayment(s, sim_year,
                                                 inputs.rpi_mean)[0])
        gross = float(s[0])
        rate_with_sl[yr] = (it_v + ni_v + sl_v) / gross * 100
        rate_without_sl[yr] = (it_v + ni_v) / gross * 100

    ax2.plot(ages, rate_with_sl, color=AMBER, linewidth=2,
             label="With Student Loan")
    ax2.plot(ages, rate_without_sl, color=SLATE, linewidth=1.5,
             linestyle="--", label="Without Student Loan")
    ax2.fill_between(ages, rate_without_sl, rate_with_sl,
                     color=AMBER, alpha=0.12)

    # Write-off marker
    ax2.axvline(ages[-1], color=RED, linewidth=1, linestyle=":", alpha=0.6)
    mid_y = (rate_with_sl[-1] + rate_without_sl[-1]) / 2
    ax2.annotate("Write-off", xy=(ages[-1], mid_y), fontsize=7,
                 color=RED, ha="right",
                 xytext=(-5, 0), textcoords="offset points")

    # Loan clearance marker (Scenario A median)
    cleared = results.a_year_loan_cleared[
        ~np.isnan(results.a_year_loan_cleared)
    ]
    if len(cleared) > 0 and results.a_loan_cleared.mean() >= 0.5:
        med_yr = float(np.median(cleared))
        clear_age = inputs.age + med_yr
        if clear_age < ages[-1]:
            yr_idx = min(int(round(med_yr)), T)
            ax2.axvline(clear_age, color=INDIGO, linewidth=1,
                        linestyle=":", alpha=0.6)
            ax2.annotate(f"Loan clears\n(age {clear_age:.0f})",
                         xy=(clear_age, rate_with_sl[yr_idx]),
                         fontsize=7, color=INDIGO,
                         xytext=(8, 8), textcoords="offset points")

    # PA taper annotation
    taper_mask = sal_path >= cfg.PA_TAPER_THRESHOLD
    if np.any(taper_mask):
        first = int(np.argmax(taper_mask))
        ax2.axvspan(ages[first], ages[-1], alpha=0.05, color=RED)
        ax2.annotate("PA taper (>\u00a3100k)",
                     xy=(ages[first], ax2.get_ylim()[1] * 0.92),
                     fontsize=7, color=RED, alpha=0.8)

    ax2.set_ylabel("Effective Deduction Rate")
    ax2.set_xlabel("Age")
    ax2.yaxis.set_major_formatter(PCT_FMT)
    ax2.set_title("Effective Total Deduction Rate Over Time",
                  fontsize=11, pad=10)
    _legend(ax2, loc="upper right")
    _add_year_axis(ax2, ages, years)

    return fig


# ═══════════════════════════════════════════════════════════════════
# Page 3 — Net Worth Comparison (THE key chart, full page)
# ═══════════════════════════════════════════════════════════════════

def _page3_net_worth(inputs: SimInputs, results: SimResults,
                     figsize=(A4W, A4H)) -> plt.Figure:
    ages = results.ages
    years = results.years

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _style(fig, ax)

    a_med = _fan(ax, ages, results.a_net_worth, INDIGO, "A: Overpay")
    b_med = _fan(ax, ages, results.b_net_worth, EMERALD, "B: Invest")

    ax.yaxis.set_major_formatter(GBP_FMT)
    ax.set_xlabel("Age")
    ax.set_ylabel("Net Worth")
    ax.set_title("Net Worth Comparison Over Time", fontsize=14, pad=15)
    _legend(ax, loc="upper left")

    # Annotate final gap
    gap = b_med[-1] - a_med[-1]
    winner_color = EMERALD if gap > 0 else INDIGO
    mid_y = (a_med[-1] + b_med[-1]) / 2
    ax.annotate(
        f"\u00a3{abs(gap):,.0f} difference",
        xy=(ages[-1], mid_y), fontsize=11, color=winner_color,
        fontweight="bold", ha="right",
        xytext=(-15, 0), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                  edgecolor=winner_color, alpha=0.9),
    )

    # Check for crossover (A overtakes B)
    cross_mask = (a_med[:-1] <= b_med[:-1]) & (a_med[1:] > b_med[1:])
    cross_idxs = np.where(cross_mask)[0]
    if len(cross_idxs) > 0:
        ci = cross_idxs[0] + 1
        ax.annotate(
            f"Overpaying overtakes\nat age {ages[ci]}",
            xy=(ages[ci], a_med[ci]), fontsize=9, color=INDIGO,
            xytext=(10, 20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=INDIGO, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                      edgecolor=INDIGO, alpha=0.9),
        )

    _add_year_axis(ax, ages, years)
    return fig


# ═══════════════════════════════════════════════════════════════════
# Page 4 — Loan Balance + Investment Pot (two fan charts)
# ═══════════════════════════════════════════════════════════════════

def _page4_loan_pot(inputs: SimInputs, results: SimResults,
                    figsize=(A4W, A4H)) -> plt.Figure:
    ages = results.ages
    years = results.years

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                   constrained_layout=True)
    _style(fig, ax1, ax2)

    # Top: loan balance
    _fan(ax1, ages, results.a_loan_balance, INDIGO, "A: Overpay")
    _fan(ax1, ages, results.b_loan_balance, EMERALD, "B: Minimum only")
    ax1.axvline(ages[-1], color=RED, linewidth=1.2, linestyle="--",
                alpha=0.6, label="Write-off")
    ax1.yaxis.set_major_formatter(GBP_FMT)
    ax1.set_ylabel("Loan Balance")
    ax1.set_xlabel("Age")
    ax1.set_title("Loan Balance Over Time", fontsize=11, pad=10)
    _legend(ax1)
    _add_year_axis(ax1, ages, years)

    # Bottom: investment pot
    _fan(ax2, ages, results.a_investment_pot, INDIGO,
         "A: Invest after clearing")
    _fan(ax2, ages, results.b_investment_pot, EMERALD,
         "B: Invest from day one")
    ax2.yaxis.set_major_formatter(GBP_FMT)
    ax2.set_ylabel("Investment Pot")
    ax2.set_xlabel("Age")
    ax2.set_title("Investment Growth", fontsize=11, pad=10)
    _legend(ax2)

    return fig


# ═══════════════════════════════════════════════════════════════════
# Web — Outcome Distribution (histogram + KDE)
# ═══════════════════════════════════════════════════════════════════

def _chart_outcome_dist(inputs: SimInputs, results: SimResults,
                        d: Dict, figsize=(WEB_W, WEB_H)) -> plt.Figure:
    """Distribution of final net worth outcomes for both strategies."""
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _style(fig, ax)

    a_final = results.a_net_worth[-1, :]
    b_final = results.b_net_worth[-1, :]

    # Bin range covering both distributions
    lo = min(a_final.min(), b_final.min())
    hi = max(a_final.max(), b_final.max())
    margin = (hi - lo) * 0.05
    bins = np.linspace(lo - margin, hi + margin, 70)

    # Histograms (light fill)
    ax.hist(a_final, bins=bins, alpha=0.18, color=INDIGO, density=True)
    ax.hist(b_final, bins=bins, alpha=0.18, color=EMERALD, density=True)

    # KDE curves
    x_range = np.linspace(lo - margin, hi + margin, 300)
    kde_a = gaussian_kde(a_final)
    kde_b = gaussian_kde(b_final)
    ya = kde_a(x_range)
    yb = kde_b(x_range)
    ax.fill_between(x_range, ya, alpha=0.12, color=INDIGO)
    ax.fill_between(x_range, yb, alpha=0.12, color=EMERALD)
    ax.plot(x_range, ya, color=INDIGO, linewidth=2.2, label="A: Overpay")
    ax.plot(x_range, yb, color=EMERALD, linewidth=2.2, label="B: Invest")

    # Median dashed lines
    med_a = np.median(a_final)
    med_b = np.median(b_final)
    ax.axvline(med_a, color=INDIGO, linewidth=1.4, linestyle="--", alpha=0.7)
    ax.axvline(med_b, color=EMERALD, linewidth=1.4, linestyle="--", alpha=0.7)

    # Annotate medians (offset if too close)
    ymax = ax.get_ylim()[1]
    y_a, y_b = 0.92, 0.80
    if abs(med_a - med_b) < (hi - lo) * 0.08:
        # Close together — stack vertically, both centered
        y_a, y_b = 0.94, 0.82
    ax.annotate(
        f"Median: {_gbp_fmt(med_a, None)}", xy=(med_a, ymax * y_a),
        fontsize=8.5, color=INDIGO, ha="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                  edgecolor=INDIGO, alpha=0.9),
    )
    ax.annotate(
        f"Median: {_gbp_fmt(med_b, None)}", xy=(med_b, ymax * y_b),
        fontsize=8.5, color=EMERALD, ha="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                  edgecolor=EMERALD, alpha=0.9),
    )

    # P(B > A) callout
    prob_b_wins = float(np.mean(b_final > a_final)) * 100
    if prob_b_wins >= 50:
        callout = f"Investing wins in\n{prob_b_wins:.0f}% of simulations"
        callout_color = EMERALD
    else:
        callout = f"Overpaying wins in\n{100 - prob_b_wins:.0f}% of simulations"
        callout_color = INDIGO
    ax.text(
        0.98, 0.97, callout, transform=ax.transAxes, fontsize=9.5,
        color=callout_color, ha="right", va="top", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG,
                  edgecolor=callout_color, alpha=0.92),
    )

    ax.xaxis.set_major_formatter(GBP_FMT)
    ax.set_xlabel("Net Worth at Write-off Age")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title("Distribution of Outcomes", fontsize=13, pad=12)
    _legend(ax, loc="upper left")

    return fig


# ═══════════════════════════════════════════════════════════════════
# Web — Breakeven Bar (simplified, no table)
# ═══════════════════════════════════════════════════════════════════

def _chart_breakeven_bar(be: BreakevenResult,
                         figsize=(WEB_W, WEB_H - 1)) -> plt.Figure:
    """Grouped bar chart showing net worth at each overpayment level."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _style(fig, ax)

    overpays = [r.monthly_overpayment for r in be.rows]
    nw_a = [r.median_nw_overpay for r in be.rows]
    nw_b = [r.median_nw_invest for r in be.rows]

    x = np.arange(len(overpays))
    w = 0.35
    ax.bar(x - w / 2, nw_a, w, color=INDIGO, label="A: Overpay NW",
           edgecolor=INDIGO_DEEP, linewidth=0.5)
    ax.bar(x + w / 2, nw_b, w, color=EMERALD, label="B: Invest NW",
           edgecolor=EMERALD_DEEP, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"\u00a3{int(o)}" for o in overpays],
                       fontsize=7.5, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(GBP_FMT)
    ax.set_xlabel("Monthly Overpayment")
    ax.set_ylabel("Median Net Worth at Write-off")
    ax.set_title("Net Worth by Overpayment Level", fontsize=13, pad=12)

    if be.breakeven_amount is not None:
        be_idx = next(i for i, o in enumerate(overpays)
                      if o == be.breakeven_amount)
        ax.annotate(
            "Breakeven", xy=(be_idx, max(nw_a[be_idx], nw_b[be_idx])),
            fontsize=9, color=AMBER, fontweight="bold",
            xytext=(0, 15), textcoords="offset points", ha="center",
            arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5),
        )
    else:
        ax.annotate(
            "Investing wins at all levels tested",
            xy=(0.5, 0.94), xycoords="axes fraction",
            fontsize=10, color=EMERALD, ha="center", fontweight="bold",
        )
    _legend(ax)

    return fig


# ═══════════════════════════════════════════════════════════════════
# Page 5 — Breakeven (grouped bar + table)
# ═══════════════════════════════════════════════════════════════════

def _page5_breakeven(be: BreakevenResult,
                     figsize=(A4W, A4H)) -> plt.Figure:
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2])
    ax_bar = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])
    _style(fig, ax_bar)

    # ── Top: grouped bar chart ──
    overpays = [r.monthly_overpayment for r in be.rows]
    nw_a = [r.median_nw_overpay for r in be.rows]
    nw_b = [r.median_nw_invest for r in be.rows]

    x = np.arange(len(overpays))
    w = 0.35
    ax_bar.bar(x - w / 2, nw_a, w, color=INDIGO, label="A: Overpay NW")
    ax_bar.bar(x + w / 2, nw_b, w, color=EMERALD, label="B: Invest NW")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"\u00a3{int(o)}" for o in overpays],
                           fontsize=7, rotation=45, ha="right")
    ax_bar.yaxis.set_major_formatter(GBP_FMT)
    ax_bar.set_xlabel("Monthly Overpayment")
    ax_bar.set_ylabel("Net Worth at Write-off")
    ax_bar.set_title("Net Worth by Overpayment Level", fontsize=11, pad=10)

    if be.breakeven_amount is not None:
        be_idx = next(i for i, o in enumerate(overpays)
                      if o == be.breakeven_amount)
        ax_bar.annotate(
            "Breakeven",
            xy=(be_idx, max(nw_a[be_idx], nw_b[be_idx])),
            fontsize=8, color=AMBER, fontweight="bold",
            xytext=(0, 12), textcoords="offset points", ha="center",
            arrowprops=dict(arrowstyle="->", color=AMBER),
        )
    else:
        ax_bar.annotate(
            "Investing wins at all levels tested",
            xy=(0.5, 0.92), xycoords="axes fraction",
            fontsize=9, color=EMERALD, ha="center", fontweight="bold",
        )
    _legend(ax_bar)

    # ── Bottom: formatted table ──
    ax_tbl.set_facecolor(BG)
    ax_tbl.axis("off")

    cols = ["Overpay/mo", "Clears", "Age", "NW (Overpay)",
            "NW (Invest)", "Winner", "Margin"]
    cell_data = []
    row_colors = []
    for r in be.rows:
        age_str = (f"{r.median_age_cleared:.0f}"
                   if r.median_age_cleared is not None else "-")
        cell_data.append([
            f"\u00a3{r.monthly_overpayment:,.0f}",
            f"{r.pct_loan_clears:.0f}%",
            age_str,
            f"\u00a3{r.median_nw_overpay:,.0f}",
            f"\u00a3{r.median_nw_invest:,.0f}",
            r.winner.title(),
            f"\u00a3{r.advantage:,.0f}",
        ])
        is_be = (be.breakeven_amount is not None
                 and r.monthly_overpayment == be.breakeven_amount)
        row_colors.append("#1a2e05" if is_be else CARD)

    table = ax_tbl.table(cellText=cell_data, colLabels=cols,
                         loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER)
        if row == 0:
            cell.set_facecolor(BG)
            cell.set_text_props(fontweight="bold", color=SLATE)
        else:
            cell.set_facecolor(row_colors[row - 1])
            color = TEXT
            if col == 5:
                win = be.rows[row - 1].winner
                color = EMERALD if win == "invest" else INDIGO
            cell.set_text_props(color=color)

    return fig


# ═══════════════════════════════════════════════════════════════════
# Lump Sum Hypothetical chart
# ═══════════════════════════════════════════════════════════════════

def _chart_lump_sum(d: Dict, figsize=(WEB_W, WEB_H - 1)) -> plt.Figure:
    """Bar chart comparing lump sum payoff vs invest over time."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _style(fig, ax)

    r = d["ls_return"]
    T = d["ls_years"]
    loan = d["ls_lump_sum"]
    annual_freed = d["ls_freed_annual"]

    # Build year-by-year trajectories
    years = np.arange(T + 1)

    # Invest lump sum: grows each year
    invest_path = loan * (1 + r) ** years

    # Pay off: freed mandatory invested as growing annuity
    payoff_path = np.zeros(T + 1)
    for t in range(1, T + 1):
        if r > 0:
            payoff_path[t] = annual_freed * (((1 + r) ** t - 1) / r)
        else:
            payoff_path[t] = annual_freed * t

    ax.plot(years, invest_path, color=EMERALD, linewidth=2.5,
            label="Invest the lump sum", solid_capstyle="round")
    ax.fill_between(years, invest_path, alpha=0.1, color=EMERALD)

    ax.plot(years, payoff_path, color=INDIGO, linewidth=2.5,
            label="Pay off + reinvest freed payments", solid_capstyle="round")
    ax.fill_between(years, payoff_path, alpha=0.1, color=INDIGO)

    # Annotate final values
    ax.annotate(
        f"\u00a3{invest_path[-1]:,.0f}",
        xy=(T, invest_path[-1]), fontsize=10, color=EMERALD,
        fontweight="bold", ha="right",
        xytext=(-10, 8), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                  edgecolor=EMERALD, alpha=0.9),
    )
    ax.annotate(
        f"\u00a3{payoff_path[-1]:,.0f}",
        xy=(T, payoff_path[-1]), fontsize=10, color=INDIGO,
        fontweight="bold", ha="right",
        xytext=(-10, -18), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                  edgecolor=INDIGO, alpha=0.9),
    )

    # Crossover point
    cross_mask = (payoff_path[:-1] <= invest_path[:-1]) & (payoff_path[1:] > invest_path[1:])
    cross_idxs = np.where(cross_mask)[0]
    if len(cross_idxs) > 0:
        ci = cross_idxs[0] + 1
        ax.annotate(
            f"Payoff overtakes\nat year {ci}",
            xy=(ci, payoff_path[ci]), fontsize=8, color=INDIGO,
            xytext=(10, 15), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=INDIGO, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                      edgecolor=INDIGO, alpha=0.9),
        )

    # Winner callout
    winner_color = EMERALD if d["ls_winner"] == "invest" else INDIGO
    winner_label = ("Investing wins" if d["ls_winner"] == "invest"
                    else "Paying off wins")
    ax.text(
        0.02, 0.97,
        f"{winner_label} by \u00a3{d['ls_advantage']:,.0f}",
        transform=ax.transAxes, fontsize=10, color=winner_color,
        fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG,
                  edgecolor=winner_color, alpha=0.92),
    )

    ax.yaxis.set_major_formatter(GBP_FMT)
    ax.set_xlabel("Years")
    ax.set_ylabel("Value")
    ax.set_title(
        f"Lump Sum Hypothetical: \u00a3{loan:,.0f} at {r * 100:.1f}% Return",
        fontsize=13, pad=12,
    )
    _legend(ax, loc="lower right")
    return fig


# ═══════════════════════════════════════════════════════════════════
# Page 6 — The Big Picture (small multiples grid)
# ═══════════════════════════════════════════════════════════════════

def _page6_big_picture(sweep: SweepResult, inputs: SimInputs,
                       figsize=(A4W, A4H)) -> plt.Figure:
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor(BG)

    # 3 rows × 2 cols = 6 cells; 5 panels + 1 legend
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    axes = [fig.add_subplot(gs[r, c]) for r, c in positions]

    n_sal = len(sweep.salaries)
    cmap = plt.cm.cool
    colors = [cmap(0.15 + 0.7 * j / max(n_sal - 1, 1))
              for j in range(n_sal)]

    # User's closest position in the grid
    loan_idx = int(np.argmin(np.abs(sweep.loan_balances
                                    - inputs.loan_balance)))
    sal_idx = int(np.argmin(np.abs(sweep.salaries - inputs.salary)))
    overpay_idx = int(np.argmin(np.abs(sweep.overpayments
                                       - inputs.monthly_overpayment)))

    for i, ax in enumerate(axes):
        _style(fig, ax)
        loan = sweep.loan_balances[i]
        for j in range(n_sal):
            lbl = (f"\u00a3{sweep.salaries[j] / 1000:.0f}k"
                   if i == 0 else None)
            ax.plot(sweep.overpayments, sweep.advantage[i, j, :],
                    color=colors[j], linewidth=1.3, label=lbl)

        ax.axhline(0, color=TEXT, linewidth=1.5, linestyle="--", alpha=0.5)
        ax.set_title(f"Loan: \u00a3{loan / 1000:.0f}k",
                     fontsize=9, color=TEXT)
        ax.yaxis.set_major_formatter(GBP_FMT)
        ax.set_xlabel("Overpay/mo (\u00a3)", fontsize=7)
        if i % 2 == 0:
            ax.set_ylabel("Invest advantage", fontsize=7)
        ax.tick_params(labelsize=6)

        # Star at user's position
        if i == loan_idx:
            user_val = sweep.advantage[loan_idx, sal_idx, overpay_idx]
            user_x = sweep.overpayments[overpay_idx]
            ax.plot(user_x, user_val, marker="*", markersize=14,
                    color=AMBER, zorder=10, markeredgecolor="white",
                    markeredgewidth=0.5)
            ax.annotate("You", (user_x, user_val),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=7, color=AMBER, fontweight="bold")

    # Legend in bottom-right cell
    ax_leg = fig.add_subplot(gs[2, 1])
    ax_leg.set_facecolor(BG)
    ax_leg.axis("off")
    for j, sal in enumerate(sweep.salaries):
        ax_leg.plot([], [], color=colors[j], linewidth=2,
                    label=f"\u00a3{sal / 1000:.0f}k salary")
    ax_leg.plot([], [], marker="*", color=AMBER, markersize=12,
                linestyle="none", label="Your position")
    ax_leg.axhline(0, color=TEXT, linewidth=1.5, linestyle="--",
                   alpha=0.5, label="Breakeven line")
    ax_leg.legend(loc="center", fontsize=8, facecolor=CARD,
                  edgecolor=BORDER, labelcolor=TEXT, ncol=2)

    fig.suptitle("The Big Picture: When Does Overpaying Win?",
                 fontsize=13, color=TEXT, fontweight="bold", y=0.99)
    return fig


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def figure_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    buf.close()
    return b64


def generate_pdf(
    inputs: SimInputs,
    results: SimResults,
    be: BreakevenResult,
    sweep: Optional[SweepResult],
    d: Dict[str, Any],
    verdict_text: str,
    path: str = "student_loan_report.pdf",
) -> str:
    """Generate the full PDF report. Returns the file path."""
    pages = [
        _page1_summary(inputs, d, verdict_text, be),
        _page2_tax(inputs, results),
        _page3_net_worth(inputs, results),
        _page4_loan_pot(inputs, results),
        _page5_breakeven(be),
        _chart_lump_sum(d, figsize=(A4W, A4H * 0.55)),
    ]
    if sweep is not None:
        pages.append(_page6_big_picture(sweep, inputs))

    with PdfPages(path) as pdf:
        for fig in pages:
            pdf.savefig(fig, facecolor=fig.get_facecolor())
    for fig in pages:
        plt.close(fig)
    return path


def get_web_charts(
    inputs: SimInputs,
    results: SimResults,
    be: BreakevenResult,
    sweep: Optional[SweepResult],
    d: Dict[str, Any],
) -> List[str]:
    """Return base64-encoded PNG chart images for web embedding.

    Returns 5 charts:
      [0] Net Worth Comparison  (fan chart — the hero chart)
      [1] Outcome Distribution  (histogram + KDE — shows risk)
      [2] Loan Balance & Investment Pot  (decomposition)
      [3] Breakeven Analysis  (grouped bar)
      [4] Lump Sum Hypothetical  (payoff vs invest trajectories)
    """
    chart_figs = [
        _page3_net_worth(inputs, results, figsize=(WEB_W, WEB_H)),
        _chart_outcome_dist(inputs, results, d, figsize=(WEB_W, WEB_H - 0.5)),
        _page4_loan_pot(inputs, results, figsize=(WEB_W, WEB_H + 2)),
        _chart_breakeven_bar(be, figsize=(WEB_W, WEB_H - 1)),
        _chart_lump_sum(d, figsize=(WEB_W, WEB_H - 1)),
    ]

    images = [figure_to_base64(f) for f in chart_figs]
    for f in chart_figs:
        plt.close(f)
    return images
