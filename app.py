"""
Flask web application for the Student Loan Plan 2 overpay-vs-invest simulator.

Single-file app using render_template_string.  Run via ``python main.py``
which starts the dev server on localhost:5000.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from flask import Flask, render_template_string, request, send_file

import numpy as np

from simulation import (
    SimInputs,
    run_simulation,
    breakeven_table,
    parameter_sweep,
)
from cli import (
    compute_display_data,
    generate_verdict_text,
    fmt,
    pct,
)
import report

app = Flask(__name__)

PDF_PATH = "student_loan_report.pdf"

# ═══════════════════════════════════════════════════════════════════
# Form parsing
# ═══════════════════════════════════════════════════════════════════

def _parse_currency(s: str) -> float:
    return float(s.replace("\u00a3", "").replace(",", "").replace(" ", ""))


def parse_form(form: dict) -> SimInputs:
    """Parse the HTML form into SimInputs."""
    return SimInputs(
        age=int(form.get("age", 25)),
        loan_balance=_parse_currency(form.get("loan_balance", "45000")),
        salary=_parse_currency(form.get("salary", "35000")),
        salary_growth_mean=float(form.get("salary_growth", "3.5")) / 100,
        years_since_first_repayment=int(form.get("years_since", "1")),
        region=form.get("region", "england"),
        monthly_overpayment=_parse_currency(form.get("overpayment", "200")),
        isa=form.get("isa", "yes") == "yes",
        inv_return_mean=float(form.get("inv_return", "7.0")) / 100,
        inv_return_std=float(form.get("inv_vol", "15.0")) / 100,
        rpi_mean=float(form.get("rpi", "3.2")) / 100,
        rpi_std=float(form.get("rpi_vol", "1.5")) / 100,
        n_iterations=int(form.get("n_iter", "10000")),
    )


# ═══════════════════════════════════════════════════════════════════
# HTML Template
# ═══════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Student Loan Plan 2: Overpay vs Invest</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  html{scroll-behavior:smooth}

  :root{
    --bg-deep:#050816;
    --bg-surface:rgba(15,23,42,0.55);
    --bg-input:rgba(8,11,22,0.85);
    --border-subtle:rgba(99,102,241,0.1);
    --border-hover:rgba(99,102,241,0.25);
    --text-primary:#f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:#64748b;
    --indigo:#818cf8;
    --indigo-deep:#6366f1;
    --violet:#8b5cf6;
    --emerald:#34d399;
    --emerald-deep:#10b981;
    --amber:#fbbf24;
    --radius-lg:16px;
    --radius-md:10px;
  }

  body{
    background:var(--bg-deep);color:var(--text-primary);
    font-family:'Inter',system-ui,-apple-system,sans-serif;
    line-height:1.6;min-height:100vh;overflow-x:hidden;
  }

  /* ── animated background mesh ── */
  .bg-mesh{
    position:fixed;inset:0;z-index:0;pointer-events:none;
    background:
      radial-gradient(ellipse 80% 50% at 15% 30%,rgba(99,102,241,0.13) 0%,transparent 70%),
      radial-gradient(ellipse 60% 40% at 85% 15%,rgba(139,92,246,0.09) 0%,transparent 70%),
      radial-gradient(ellipse 50% 60% at 55% 85%,rgba(16,185,129,0.07) 0%,transparent 70%);
    animation:meshDrift 25s ease-in-out infinite alternate;
  }
  @keyframes meshDrift{
    0%{opacity:1;transform:scale(1) translate(0,0)}
    50%{opacity:.85;transform:scale(1.06) translate(-1%,2%)}
    100%{opacity:1;transform:scale(1) translate(1%,-1%)}
  }

  .container{max-width:1140px;margin:0 auto;padding:2rem 1.5rem;position:relative;z-index:1}

  /* ── hero header ── */
  .hero{text-align:center;padding:1.5rem 0 2.5rem}
  .hero-pill{
    display:inline-flex;align-items:center;gap:.45rem;
    background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.18);
    border-radius:100px;padding:.3rem 1rem;font-size:.75rem;
    color:var(--indigo);font-weight:600;margin-bottom:1rem;
    letter-spacing:.06em;text-transform:uppercase;
  }
  .hero-dot{width:6px;height:6px;background:var(--indigo);border-radius:50%;animation:dotPulse 2s ease-in-out infinite}
  @keyframes dotPulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}
  .hero h1{
    font-size:clamp(1.5rem,4vw,2.5rem);font-weight:800;
    letter-spacing:-.035em;line-height:1.15;
    background:linear-gradient(135deg,#e2e8f0 0%,#818cf8 45%,#34d399 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;
  }
  .hero-sub{color:var(--text-secondary);margin-top:.6rem;font-size:.92rem;font-weight:400}

  /* ── glass cards ── */
  .card{
    background:var(--bg-surface);
    backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
    border:1px solid var(--border-subtle);
    border-radius:var(--radius-lg);padding:1.8rem;
    margin-bottom:1.4rem;position:relative;overflow:hidden;
    transition:border-color .3s,box-shadow .3s,transform .3s;
  }
  .card::before{
    content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent 5%,rgba(99,102,241,.25) 50%,transparent 95%);
  }
  .card:hover{border-color:var(--border-hover);box-shadow:0 8px 40px rgba(99,102,241,.06)}

  /* winner glow effects */
  .winner-glow-emerald{
    border-color:rgba(52,211,153,.35) !important;
    box-shadow:0 0 40px rgba(16,185,129,.08),0 8px 32px rgba(16,185,129,.05) !important;
  }
  .winner-glow-emerald::before{background:linear-gradient(90deg,transparent 5%,rgba(52,211,153,.5) 50%,transparent 95%) !important}
  .winner-glow-indigo{
    border-color:rgba(129,140,248,.35) !important;
    box-shadow:0 0 40px rgba(99,102,241,.08),0 8px 32px rgba(99,102,241,.05) !important;
  }
  .winner-glow-indigo::before{background:linear-gradient(90deg,transparent 5%,rgba(129,140,248,.5) 50%,transparent 95%) !important}

  /* section headers with accent */
  .sh{display:flex;align-items:center;gap:.65rem;margin-bottom:1.2rem}
  .sh-icon{
    width:34px;height:34px;border-radius:9px;
    display:flex;align-items:center;justify-content:center;
    font-size:.85rem;flex-shrink:0;font-weight:700;
  }
  .sh-icon.i-blue{background:rgba(59,130,246,.12);color:#60a5fa}
  .sh-icon.i-slate{background:rgba(148,163,184,.1);color:#94a3b8}
  .sh-icon.i-indigo{background:rgba(129,140,248,.12);color:var(--indigo)}
  .sh-icon.i-emerald{background:rgba(52,211,153,.12);color:var(--emerald)}
  .sh-icon.i-amber{background:rgba(251,191,36,.12);color:var(--amber)}
  .sh-icon.i-violet{background:rgba(139,92,246,.12);color:var(--violet)}

  h2{font-size:1.1rem;font-weight:700;color:var(--text-primary);letter-spacing:-.015em}

  /* ── form ── */
  .form-grid{
    display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
    gap:1rem 1.5rem;
  }
  .form-group{display:flex;flex-direction:column}
  .form-group label{
    font-size:.78rem;color:var(--text-secondary);margin-bottom:.3rem;
    font-weight:500;letter-spacing:.02em;
  }
  .form-group input,.form-group select{
    background:var(--bg-input);
    border:1px solid rgba(71,85,105,.35);border-radius:var(--radius-md);
    color:var(--text-primary);padding:.6rem .85rem;font-size:.88rem;
    font-family:inherit;transition:border-color .25s,box-shadow .25s;
  }
  .form-group input:hover,.form-group select:hover{border-color:rgba(71,85,105,.55)}
  .form-group input:focus,.form-group select:focus{
    outline:none;border-color:var(--indigo-deep);
    box-shadow:0 0 0 3px rgba(99,102,241,.12),0 0 24px rgba(99,102,241,.04);
  }

  /* ── buttons ── */
  .btn{
    display:inline-flex;align-items:center;justify-content:center;gap:.5rem;
    padding:.75rem 2rem;border:none;border-radius:var(--radius-md);
    font-size:.95rem;font-weight:600;cursor:pointer;font-family:inherit;
    text-decoration:none;position:relative;overflow:hidden;
    transition:transform .2s,box-shadow .3s;
  }
  .btn::after{
    content:'';position:absolute;inset:0;
    background:linear-gradient(rgba(255,255,255,.12),transparent);
    opacity:0;transition:opacity .25s;
  }
  .btn:hover::after{opacity:1}
  .btn:hover{transform:translateY(-2px)}
  .btn:active{transform:translateY(0) scale(.98)}
  .btn:disabled{opacity:.4;cursor:not-allowed;transform:none}

  .btn-primary{
    background:linear-gradient(135deg,var(--indigo-deep),var(--violet));
    color:#fff;box-shadow:0 4px 20px rgba(99,102,241,.3);
  }
  .btn-primary:hover{box-shadow:0 8px 30px rgba(99,102,241,.4)}
  .btn-success{
    background:linear-gradient(135deg,var(--emerald-deep),var(--emerald));
    color:#fff;box-shadow:0 4px 20px rgba(16,185,129,.25);
  }
  .btn-success:hover{box-shadow:0 8px 30px rgba(16,185,129,.35)}

  /* ── options grid ── */
  .options-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.4rem;margin-bottom:1.4rem}
  @media(max-width:768px){.options-grid{grid-template-columns:1fr}}

  /* ── stat rows ── */
  .stat-row{
    display:flex;justify-content:space-between;align-items:center;
    padding:.5rem 0;border-bottom:1px solid rgba(51,65,85,.3);
    transition:background .2s,padding .2s;
  }
  .stat-row:last-child{border-bottom:none}
  .stat-row:hover{background:rgba(99,102,241,.025);padding-left:.4rem;padding-right:.4rem;border-radius:6px}
  .stat-label{color:var(--text-secondary);font-size:.86rem}
  .stat-value{font-weight:600;font-size:.86rem;font-variant-numeric:tabular-nums}

  .option-a .stat-value{color:var(--indigo)}
  .option-b .stat-value{color:var(--emerald)}

  .tag-overpay{color:var(--indigo)}
  .tag-invest{color:var(--emerald)}
  .tag-amber{color:var(--amber)}

  /* ── freed cashflow callout ── */
  .freed-cashflow{
    background:rgba(129,140,248,.06);border:1px solid rgba(129,140,248,.18);
    border-radius:var(--radius-md);padding:.8rem 1rem;margin:.7rem 0;
  }
  .freed-header{
    font-size:.78rem;font-weight:700;color:var(--indigo);
    text-transform:uppercase;letter-spacing:.04em;margin-bottom:.55rem;
  }
  .freed-grid{display:grid;grid-template-columns:1fr 1fr;gap:.3rem .8rem}
  .freed-item{display:flex;justify-content:space-between;align-items:center;padding:.15rem 0}
  .freed-label{font-size:.82rem;color:var(--text-secondary)}
  .freed-val{font-size:.82rem;font-weight:600;color:var(--indigo);font-variant-numeric:tabular-nums}
  .freed-total{
    grid-column:1/-1;border-top:1px solid rgba(129,140,248,.15);
    padding-top:.35rem;margin-top:.15rem;
  }
  .freed-total .freed-val{font-size:.9rem;color:var(--text-primary)}

  /* ── warning box ── */
  .warning{
    background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.18);
    border-radius:var(--radius-md);padding:.75rem 1rem;margin-top:.8rem;
    font-size:.84rem;color:#fcd34d;display:flex;gap:.5rem;line-height:1.55;
  }
  .warning-icon{flex-shrink:0;font-size:1rem;line-height:1.55}

  /* ── verdict ── */
  .verdict-card{position:relative}
  .verdict-glow{
    position:absolute;top:-30%;right:-10%;width:280px;height:280px;
    border-radius:50%;filter:blur(80px);opacity:.12;pointer-events:none;
  }
  .verdict-glow.glow-emerald{background:var(--emerald-deep)}
  .verdict-glow.glow-indigo{background:var(--indigo-deep)}
  .verdict-winner{
    font-size:clamp(1.15rem,3vw,1.5rem);font-weight:800;
    letter-spacing:-.02em;margin-bottom:.15rem;position:relative;
  }
  .verdict-by{font-size:.95rem;color:var(--text-secondary);margin-bottom:.9rem;font-weight:500}
  .verdict-text{color:var(--text-secondary);line-height:1.75;font-size:.9rem;position:relative}

  /* ── breakeven table ── */
  .table-wrap{overflow-x:auto;margin-top:.5rem;border-radius:var(--radius-md);border:1px solid rgba(51,65,85,.25)}
  .be-table{width:100%;border-collapse:collapse;font-size:.84rem}
  .be-table th{
    text-align:left;padding:.65rem .8rem;
    background:rgba(15,23,42,.45);
    color:var(--text-secondary);font-weight:600;font-size:.76rem;
    text-transform:uppercase;letter-spacing:.05em;
    border-bottom:1px solid rgba(51,65,85,.25);
  }
  .be-table td{
    padding:.5rem .8rem;border-bottom:1px solid rgba(51,65,85,.15);
    transition:background .15s;
  }
  .be-table tbody tr:hover td{background:rgba(99,102,241,.03)}
  .be-table .breakeven-row{background:rgba(16,185,129,.07)}
  .be-table .breakeven-row td{font-weight:600;border-bottom-color:rgba(16,185,129,.15)}
  .be-table .winner-invest{color:var(--emerald);font-weight:600}
  .be-table .winner-overpay{color:var(--indigo);font-weight:600}

  /* ── lump sum hypothetical ── */
  .ls-option{
    background:rgba(15,23,42,.35);border:1px solid rgba(51,65,85,.25);
    border-radius:var(--radius-md);padding:1rem;
  }
  .ls-option.ls-winner{
    border-color:rgba(99,102,241,.2);
    box-shadow:0 0 20px rgba(99,102,241,.04);
  }
  .ls-option-header{
    display:flex;align-items:center;gap:.5rem;
    font-size:.88rem;font-weight:700;color:var(--text-primary);margin-bottom:.7rem;
  }
  .ls-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
  .ls-verdict{
    text-align:center;padding:.9rem 0 .2rem;
    font-size:.92rem;color:var(--text-secondary);font-weight:500;
  }
  .ls-verdict-label{font-weight:700;font-size:.95rem}

  /* ── invest-favorable threshold ── */
  .invest-threshold{
    display:flex;gap:.85rem;align-items:flex-start;
    background:rgba(52,211,153,.06);border:1px solid rgba(52,211,153,.2);
    border-radius:var(--radius-md);padding:.9rem 1rem;margin-top:1rem;
  }
  .invest-threshold-icon{
    flex-shrink:0;width:34px;height:34px;border-radius:9px;
    background:rgba(52,211,153,.12);color:var(--emerald);
    display:flex;align-items:center;justify-content:center;
  }
  .invest-threshold-title{
    font-size:.92rem;font-weight:700;color:var(--emerald);margin-bottom:.25rem;
  }
  .invest-threshold-detail{
    font-size:.84rem;color:var(--text-secondary);line-height:1.6;
  }
  .invest-threshold-detail strong{color:var(--text-primary)}

  .tip{
    background:rgba(251,191,36,.05);border-left:3px solid var(--amber);
    padding:.75rem 1rem;margin-top:1rem;font-size:.84rem;color:#fcd34d;
    border-radius:0 var(--radius-md) var(--radius-md) 0;line-height:1.6;
  }
  .summary-line{margin-top:.8rem;font-size:.92rem;color:var(--text-primary)}

  /* ── charts ── */
  .chart-img{width:100%;border-radius:var(--radius-md);margin-bottom:.5rem;border:1px solid rgba(51,65,85,.15)}
  .chart-desc{color:var(--text-muted);font-size:.82rem;margin-bottom:.8rem;line-height:1.55}

  /* ── loading overlay ── */
  #loading{
    display:none;position:fixed;inset:0;
    background:rgba(5,8,22,.92);backdrop-filter:blur(10px);z-index:999;
    justify-content:center;align-items:center;flex-direction:column;gap:1.2rem;
  }
  .loader{position:relative;width:56px;height:56px}
  .loader-ring{
    position:absolute;inset:0;border-radius:50%;
    border:3px solid transparent;
  }
  .loader-ring:nth-child(1){border-top-color:var(--indigo);animation:spin 1s linear infinite}
  .loader-ring:nth-child(2){border-right-color:var(--violet);animation:spin 1.5s linear infinite reverse;inset:6px}
  .loader-ring:nth-child(3){border-bottom-color:var(--emerald-deep);animation:spin 2s linear infinite;inset:12px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .load-title{color:var(--text-primary);font-size:.95rem;font-weight:600}
  .load-sub{color:var(--text-muted);font-size:.82rem;animation:fadeIO 2.5s ease-in-out infinite}
  @keyframes fadeIO{0%,100%{opacity:.35}50%{opacity:1}}

  /* ── reveal on scroll ── */
  .reveal{opacity:0;transform:translateY(24px);transition:opacity .55s cubic-bezier(.4,0,.2,1),transform .55s cubic-bezier(.4,0,.2,1)}
  .reveal.visible{opacity:1;transform:translateY(0)}

  /* stagger delays via inline style --d */
  .reveal{transition-delay:calc(var(--d,0) * 80ms)}

  /* ── download ── */
  .dl-section{text-align:center;padding:1.5rem 0 2rem}

  /* ── about section ── */
  .about-toggle{
    display:flex;align-items:center;justify-content:space-between;
    cursor:pointer;user-select:none;width:100%;background:none;border:none;
    color:inherit;font:inherit;padding:0;text-align:left;
  }
  .about-toggle:focus-visible{outline:2px solid var(--indigo);outline-offset:4px;border-radius:4px}
  .about-chevron{
    width:22px;height:22px;color:var(--text-muted);transition:transform .35s cubic-bezier(.4,0,.2,1);flex-shrink:0;
  }
  .about-toggle[aria-expanded="true"] .about-chevron{transform:rotate(180deg)}
  .about-body{
    max-height:0;overflow:hidden;
    transition:max-height .45s cubic-bezier(.4,0,.2,1),opacity .35s ease;
    opacity:0;
  }
  .about-body.open{max-height:2200px;opacity:1;margin-top:1.2rem}
  .about-body p,.about-body li{color:var(--text-secondary);font-size:.88rem;line-height:1.75}
  .about-body p{margin-bottom:.8rem}
  .about-body h3{
    font-size:.92rem;font-weight:700;color:var(--text-primary);
    margin:1.4rem 0 .5rem;letter-spacing:-.01em;
  }
  .about-body h3:first-child{margin-top:.2rem}
  .about-body ul{list-style:none;padding:0;margin:0 0 .8rem}
  .about-body li{padding:.25rem 0 .25rem 1.2rem;position:relative}
  .about-body li::before{
    content:'';position:absolute;left:0;top:.65rem;
    width:6px;height:6px;border-radius:50%;
    background:linear-gradient(135deg,var(--indigo),var(--emerald));
  }
  .about-body code{
    background:rgba(99,102,241,.1);color:var(--indigo);
    padding:.1rem .4rem;border-radius:4px;font-size:.82rem;
  }

  /* ── footer ── */
  .footer{text-align:center;padding:1rem 0 2rem;color:var(--text-muted);font-size:.78rem}

  /* ── responsive ── */
  @media(max-width:640px){
    .container{padding:1rem}
    .card{padding:1.2rem}
    .form-grid{grid-template-columns:1fr}
    .hero h1{font-size:1.35rem}
  }
  @media(prefers-reduced-motion:reduce){
    *{animation-duration:.01ms !important;transition-duration:.01ms !important}
    .bg-mesh{animation:none}
    .reveal{opacity:1;transform:none}
  }
</style>
</head>
<body>
<div class="bg-mesh"></div>

<!-- Loading overlay -->
<div id="loading">
  <div class="loader">
    <div class="loader-ring"></div>
    <div class="loader-ring"></div>
    <div class="loader-ring"></div>
  </div>
  <p class="load-title">Running Monte Carlo simulation</p>
  <p class="load-sub">Crunching 10,000+ scenarios...</p>
</div>

<div class="container">

<!-- Hero -->
<header class="hero">
  <div class="hero-pill"><span class="hero-dot"></span> Monte Carlo Simulator</div>
  <h1>Student Loan Plan 2<br>Overpay vs Invest</h1>
  <p class="hero-sub">Should you overpay your student loan or invest the money instead?</p>
</header>

<!-- About -->
<div class="card">
  <button class="about-toggle" id="about-toggle" aria-expanded="false" aria-controls="about-body">
    <div class="sh" style="margin-bottom:0">
      <div class="sh-icon i-violet">
        <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
      </div>
      <h2>About This Tool</h2>
    </div>
    <svg class="about-chevron" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"/></svg>
  </button>

  <div class="about-body" id="about-body">

    <h3>Intention</h3>
    <p>
      Determines whether a UK Plan 2 student loan borrower is better off overpaying their loan
      or investing the money instead, using Monte Carlo simulation to account for uncertainty
      in market returns, inflation, and salary growth.
    </p>

    <h3>Core Functions</h3>
    <ul>
      <li><code>run_simulation()</code> &mdash; runs N stochastic iterations comparing overpay vs invest, outputs per-year loan balance, investment pot, and net worth arrays for both strategies</li>
      <li><code>breakeven_table()</code> &mdash; tests overpayment levels (&pound;0&ndash;&pound;1,000/mo) to find the threshold where overpaying beats investing</li>
      <li><code>parameter_sweep()</code> &mdash; 3D sensitivity analysis across loan balances, salaries, and overpayment amounts (210 combinations)</li>
      <li><code>compute_display_data()</code> &mdash; extracts 30+ display metrics from simulation results (clearance rates, net worth, lifetime payments, etc.)</li>
      <li><code>generate_verdict_text()</code> &mdash; produces a plain-English summary of the winner and why</li>
      <li><code>generate_pdf()</code> / <code>get_web_charts()</code> &mdash; renders charts and a downloadable PDF report</li>
    </ul>

    <h3>Tax &amp; Loan Functions (<code>tax.py</code>)</h3>
    <ul>
      <li><code>income_tax()</code> &mdash; England/Scotland band calculations with &pound;100k personal allowance taper</li>
      <li><code>national_insurance()</code> &mdash; Class 1 employee NI</li>
      <li><code>student_loan_repayment()</code> &mdash; 9% above Plan 2 threshold</li>
      <li><code>student_loan_interest_rate()</code> &mdash; RPI-linked rate varying by salary band</li>
      <li><code>take_home_pay()</code> &mdash; gross to net</li>
      <li><code>marginal_rate_breakdown()</code> &mdash; combined IT + NI + SL marginal rates</li>
      <li><code>investment_tax()</code> &mdash; CGT and dividend tax for GIA holdings</li>
    </ul>

    <h3>Simulation Assumptions</h3>
    <ul>
      <li>Investment returns: AR(1) mean-reversion, correlated with inflation (0.3)</li>
      <li>3% annual redundancy/salary setback probability</li>
      <li>Plan 2 write-off at 30 years, 2025/26 threshold freeze schedule</li>
      <li>GIA portfolio split: 70% capital gains / 30% dividends</li>
    </ul>

    <p style="color:var(--text-muted);font-size:.8rem;margin-top:1rem">
      For educational/illustrative purposes only. Not financial advice.
    </p>

  </div>
</div>

<!-- Input Form -->
<div class="card">
  <div class="sh">
    <div class="sh-icon i-blue">
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
    </div>
    <h2>Your Details</h2>
  </div>
  <form method="POST" id="sim-form">
    <div class="form-grid">
      <div class="form-group">
        <label>Current age</label>
        <input type="number" name="age" value="{{ form.age or 25 }}" min="18" max="65">
      </div>
      <div class="form-group">
        <label>Starting loan balance</label>
        <input type="text" name="loan_balance" value="{{ form.loan_balance or '45000' }}">
      </div>
      <div class="form-group">
        <label>Current annual gross salary</label>
        <input type="text" name="salary" value="{{ form.salary or '35000' }}">
      </div>
      <div class="form-group">
        <label>Expected salary growth %/yr</label>
        <input type="number" step="0.1" name="salary_growth" value="{{ form.salary_growth or '3.5' }}">
      </div>
      <div class="form-group">
        <label>Years since first repayment</label>
        <input type="number" name="years_since" value="{{ form.years_since or 1 }}" min="0" max="29">
      </div>
      <div class="form-group">
        <label>Region</label>
        <select name="region">
          <option value="england" {{ 'selected' if (form.region or 'england')=='england' }}>England</option>
          <option value="scotland" {{ 'selected' if form.region=='scotland' }}>Scotland</option>
        </select>
      </div>
      <div class="form-group">
        <label>Monthly overpayment amount</label>
        <input type="text" name="overpayment" value="{{ form.overpayment or '200' }}">
      </div>
      <div class="form-group">
        <label>Invest in ISA?</label>
        <select name="isa">
          <option value="yes" {{ 'selected' if (form.isa or 'yes')=='yes' }}>Yes</option>
          <option value="no" {{ 'selected' if form.isa=='no' }}>No</option>
        </select>
      </div>
      <div class="form-group">
        <label>Expected annual investment return %</label>
        <input type="number" step="0.1" name="inv_return" value="{{ form.inv_return or '7.0' }}">
      </div>
      <div class="form-group">
        <label>Return volatility / std dev %</label>
        <input type="number" step="0.1" name="inv_vol" value="{{ form.inv_vol or '15.0' }}">
      </div>
      <div class="form-group">
        <label>Average RPI %</label>
        <input type="number" step="0.1" name="rpi" value="{{ form.rpi or '3.2' }}">
      </div>
      <div class="form-group">
        <label>RPI volatility %</label>
        <input type="number" step="0.1" name="rpi_vol" value="{{ form.rpi_vol or '1.5' }}">
      </div>
      <div class="form-group">
        <label>Number of iterations</label>
        <input type="number" name="n_iter" value="{{ form.n_iter or '10000' }}" min="100" max="100000">
      </div>
    </div>
    <div style="margin-top:1.2rem">
      <button type="submit" class="btn btn-primary" id="submit-btn">
        <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
        Run Simulation
      </button>
    </div>
  </form>
</div>

{% if d %}
<!-- ═══════════════════════════════════════════════════════════ -->
<!-- RESULTS                                                     -->
<!-- ═══════════════════════════════════════════════════════════ -->

<!-- Section 1: You Now -->
<div class="card reveal" style="--d:0">
  <div class="sh">
    <div class="sh-icon i-slate">
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
    </div>
    <h2>You Now</h2>
  </div>
  <div class="stat-row"><span class="stat-label">Age</span><span class="stat-value">{{ d.age }}</span></div>
  <div class="stat-row"><span class="stat-label">Salary</span><span class="stat-value">{{ fmt(d.salary) }}</span></div>
  <div class="stat-row"><span class="stat-label">Loan balance</span><span class="stat-value">{{ fmt(d.loan_balance) }}</span></div>
  <div class="stat-row"><span class="stat-label">Take-home pay (annual)</span><span class="stat-value">{{ fmt(d.annual_take_home) }}</span></div>
  <div class="stat-row"><span class="stat-label">Take-home pay (monthly)</span><span class="stat-value">{{ fmt(d.monthly_take_home) }}</span></div>
  <div class="stat-row">
    <span class="stat-label">Marginal rate</span>
    <span class="stat-value tag-amber">
      {{ pct(d.marginal.total_marginal_pct) }}
      ({{ pct(d.marginal.income_tax_pct) }} IT + {{ pct(d.marginal.ni_pct) }} NI + {{ pct(d.marginal.sl_pct) }} SL)
    </span>
  </div>
  <div class="stat-row"><span class="stat-label">Monthly mandatory SL cost</span><span class="stat-value">{{ fmt(d.monthly_mandatory) }}</span></div>
  <div class="stat-row"><span class="stat-label">Loan interest rate</span><span class="stat-value">{{ pct(d.sl_rate * 100) }}</span></div>
  <div class="stat-row"><span class="stat-label">Years until write-off</span><span class="stat-value">{{ d.remaining_years }}</span></div>
  <div class="stat-row"><span class="stat-label">Age at write-off</span><span class="stat-value">{{ d.write_off_age }}</span></div>
</div>

<!-- Options A & B side by side -->
<div class="options-grid">

  <!-- Section 2: Option A -->
  <div class="card option-a reveal {{ 'winner-glow-indigo' if d.winner == 'overpay' }}" style="--d:1">
    <div class="sh">
      <div class="sh-icon i-indigo">A</div>
      <h2 class="tag-overpay">Overpay {{ fmt(d.overpayment) }}/mo</h2>
    </div>
    {% if d.a_med_age is not none %}
    <div class="stat-row"><span class="stat-label">Loan clears at age (median)</span><span class="stat-value">{{ "%.0f"|format(d.a_med_age) }}</span></div>
    {% else %}
    <div class="stat-row"><span class="stat-label">Loan clears at age (median)</span><span class="stat-value">Does not clear</span></div>
    {% endif %}
    <div class="stat-row"><span class="stat-label">Simulations where loan clears</span><span class="stat-value">{{ "%.0f"|format(d.a_pct_cleared) }}%</span></div>

    {% if d.a_years_investing > 0 %}
    <!-- Freed cashflow reinvestment callout -->
    <div class="freed-cashflow">
      <div class="freed-header">Freed Cashflow Reinvested After Loan Clears</div>
      <div class="freed-grid">
        <div class="freed-item">
          <span class="freed-label">Overpayment freed</span>
          <span class="freed-val">{{ fmt(d.a_freed_overpayment) }}/mo</span>
        </div>
        <div class="freed-item">
          <span class="freed-label">Mandatory SL freed</span>
          <span class="freed-val">{{ fmt(d.a_freed_mandatory) }}/mo</span>
        </div>
        <div class="freed-item freed-total">
          <span class="freed-label">Total reinvested</span>
          <span class="freed-val">{{ fmt(d.a_monthly_invest) }}/mo</span>
        </div>
        <div class="freed-item">
          <span class="freed-label">Duration</span>
          <span class="freed-val">{{ "%.0f"|format(d.a_years_investing) }} years</span>
        </div>
        <div class="freed-item">
          <span class="freed-label">Total contributions</span>
          <span class="freed-val">{{ fmt(d.a_total_reinvested) }}</span>
        </div>
      </div>
    </div>
    {% else %}
    <div class="stat-row"><span class="stat-label">Freed cashflow reinvested</span><span class="stat-value" style="color:var(--text-muted)">None &mdash; loan does not clear</span></div>
    {% endif %}

    <div class="stat-row"><span class="stat-label">Net worth at write-off (median)</span><span class="stat-value">{{ fmt(d.a_nw) }}</span></div>
    <div class="stat-row"><span class="stat-label">Total lifetime payments to SLC</span><span class="stat-value">{{ fmt(d.a_paid) }}</span></div>
    {% if d.a_pct_cleared < 60 %}
    <div class="warning">
      <span class="warning-icon">&#9888;</span>
      <span>In {{ "%.0f"|format(100 - d.a_pct_cleared) }}% of scenarios your overpayments don't clear the loan before
      write-off &mdash; that money would be wasted on a debt that gets forgiven anyway.</span>
    </div>
    {% endif %}
  </div>

  <!-- Section 3: Option B -->
  <div class="card option-b reveal {{ 'winner-glow-emerald' if d.winner == 'invest' }}" style="--d:2">
    <div class="sh">
      <div class="sh-icon i-emerald">B</div>
      <h2 class="tag-invest">Invest {{ fmt(d.overpayment) }}/mo Instead</h2>
    </div>
    <div class="stat-row"><span class="stat-label">Invest per month</span><span class="stat-value">{{ fmt(d.b_monthly_invest) }}</span></div>
    <div class="stat-row"><span class="stat-label">Investment duration</span><span class="stat-value">{{ d.remaining_years }} years (full term)</span></div>
    <div class="stat-row"><span class="stat-label">Loan written off at age</span><span class="stat-value">{{ d.write_off_age }}</span></div>
    <div class="stat-row"><span class="stat-label">Amount forgiven for free</span><span class="stat-value">{{ fmt(d.b_forgiven) }}</span></div>
    <div class="stat-row"><span class="stat-label">Net worth at write-off (median)</span><span class="stat-value">{{ fmt(d.b_nw) }}</span></div>
    <div class="stat-row"><span class="stat-label">Total lifetime payments to SLC</span><span class="stat-value">{{ fmt(d.b_paid) }}</span></div>
  </div>
</div>

<!-- Section 4: The Verdict -->
<div class="card verdict-card reveal" style="--d:3">
  <div class="verdict-glow {{ 'glow-emerald' if d.winner == 'invest' else 'glow-indigo' }}"></div>
  <div class="sh">
    <div class="sh-icon {{ 'i-emerald' if d.winner == 'invest' else 'i-indigo' }}">
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
    </div>
    <h2>The Verdict</h2>
  </div>
  <div class="verdict-winner {{ 'tag-invest' if d.winner == 'invest' else 'tag-overpay' }}">
    {{ "INVESTING WINS" if d.winner == "invest" else "OVERPAYING WINS" }}
  </div>
  <div class="verdict-by">by {{ fmt(d.adv_abs) }} ({{ pct(d.adv_pct) }})</div>
  <p class="verdict-text">{{ verdict_text }}</p>
</div>

<!-- Section 5: Breakeven Table -->
<div class="card reveal" style="--d:4">
  <div class="sh">
    <div class="sh-icon i-amber">
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 6h18M3 12h18M3 18h18"/></svg>
    </div>
    <h2>What Overpayment Would Be Worthwhile?</h2>
  </div>
  <div class="table-wrap">
    <table class="be-table">
      <thead>
        <tr>
          <th>Overpay/mo</th>
          <th>Clears</th>
          <th>Age</th>
          <th>Winner</th>
          <th>By</th>
        </tr>
      </thead>
      <tbody>
        {% for r in be.rows %}
        <tr class="{{ 'breakeven-row' if be.breakeven_amount is not none and r.monthly_overpayment == be.breakeven_amount }}">
          <td>{{ fmt(r.monthly_overpayment) }}</td>
          <td>{{ "%.0f"|format(r.pct_loan_clears) }}%</td>
          <td>{{ "%.0f"|format(r.median_age_cleared) if r.median_age_cleared is not none else "-" }}</td>
          <td class="{{ 'winner-invest' if r.winner == 'invest' else 'winner-overpay' }}">{{ r.winner }}</td>
          <td>{{ fmt(r.advantage) }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- Invest-favorable threshold callout -->
  <div class="invest-threshold">
    <div class="invest-threshold-icon">
      <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg>
    </div>
    <div class="invest-threshold-body">
      {% if d.invest_favorable_below is not none %}
        <div class="invest-threshold-title">Investing is favorable below {{ fmt(d.invest_favorable_below) }}/mo</div>
        <div class="invest-threshold-detail">
          At any overpayment under <strong>{{ fmt(d.invest_favorable_below) }}/mo</strong>, investing the money instead
          leaves you better off at write-off. You'd need to overpay
          <strong>{{ fmt(d.invest_favorable_below) }}/mo+</strong> to beat investing.
        </div>
      {% else %}
        <div class="invest-threshold-title">Investing is favorable at every level tested</div>
        <div class="invest-threshold-detail">
          Even at <strong>{{ fmt(d.invest_favorable_max_tested) }}/mo</strong>, investing still wins.
          No tested overpayment amount beats investing &mdash; your loan will likely be written off regardless.
        </div>
      {% endif %}
    </div>
  </div>

  <div class="tip">
    <strong>Tip:</strong> Salary sacrifice into your pension reduces both income tax and student loan
    repayments, effectively cutting your marginal rate. Consider maximising employer contributions
    before overpaying your loan.
  </div>
</div>

<!-- Lump Sum Hypothetical -->
<div class="card reveal" style="--d:5">
  <div class="sh">
    <div class="sh-icon i-violet">
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z"/></svg>
    </div>
    <h2>Hypothetical: Lump Sum Payoff</h2>
  </div>
  <p style="color:var(--text-secondary);font-size:.86rem;margin-bottom:1rem;line-height:1.6">
    What if you had <strong style="color:var(--text-primary)">{{ fmt(d.ls_lump_sum) }}</strong> in cash right now?
    Should you pay off the entire loan or invest the lump sum instead?
    <span style="color:var(--text-muted)">(Deterministic estimate at {{ pct(d.ls_return * 100) }} return &mdash; no Monte Carlo.)</span>
  </p>

  <div class="options-grid" style="margin-bottom:0">
    <!-- Pay off -->
    <div class="ls-option {{ 'ls-winner' if d.ls_winner == 'payoff' }}">
      <div class="ls-option-header">
        <span class="ls-dot" style="background:var(--indigo)"></span>
        Pay Off Now
      </div>
      <div class="stat-row"><span class="stat-label">Spend today</span><span class="stat-value" style="color:var(--indigo)">{{ fmt(d.ls_lump_sum) }}</span></div>
      <div class="stat-row"><span class="stat-label">Mandatory SL freed</span><span class="stat-value" style="color:var(--indigo)">{{ fmt(d.ls_freed_monthly) }}/mo</span></div>
      <div class="stat-row"><span class="stat-label">Freed payments reinvested for</span><span class="stat-value" style="color:var(--indigo)">{{ d.ls_years }} years</span></div>
      <div class="stat-row"><span class="stat-label">Total mandatory saved</span><span class="stat-value" style="color:var(--indigo)">{{ fmt(d.ls_total_mandatory_saved) }}</span></div>
      <div class="stat-row" style="border-top:1px solid rgba(51,65,85,.3);padding-top:.6rem;margin-top:.3rem">
        <span class="stat-label" style="font-weight:600;color:var(--text-primary)">Net worth at write-off age</span>
        <span class="stat-value" style="color:var(--indigo);font-size:.95rem">{{ fmt(d.ls_payoff_fv) }}</span>
      </div>
    </div>

    <!-- Invest lump sum -->
    <div class="ls-option {{ 'ls-winner' if d.ls_winner == 'invest' }}">
      <div class="ls-option-header">
        <span class="ls-dot" style="background:var(--emerald)"></span>
        Invest the Lump Sum
      </div>
      <div class="stat-row"><span class="stat-label">Invest today</span><span class="stat-value" style="color:var(--emerald)">{{ fmt(d.ls_lump_sum) }}</span></div>
      <div class="stat-row"><span class="stat-label">Grows for</span><span class="stat-value" style="color:var(--emerald)">{{ d.ls_years }} years at {{ pct(d.ls_return * 100) }}</span></div>
      <div class="stat-row"><span class="stat-label">Mandatory SL payments continue</span><span class="stat-value" style="color:var(--emerald)">{{ fmt(d.ls_freed_monthly) }}/mo</span></div>
      <div class="stat-row"><span class="stat-label">Loan written off for free</span><span class="stat-value" style="color:var(--emerald)">Yes</span></div>
      <div class="stat-row" style="border-top:1px solid rgba(51,65,85,.3);padding-top:.6rem;margin-top:.3rem">
        <span class="stat-label" style="font-weight:600;color:var(--text-primary)">Net worth at write-off age</span>
        <span class="stat-value" style="color:var(--emerald);font-size:.95rem">{{ fmt(d.ls_invest_fv) }}</span>
      </div>
    </div>
  </div>

  <div class="ls-verdict">
    <span class="ls-verdict-label {{ 'tag-invest' if d.ls_winner == 'invest' else 'tag-overpay' }}">
      {{ "Investing the lump sum wins" if d.ls_winner == "invest" else "Paying off wins" }}
    </span>
    by {{ fmt(d.ls_advantage) }} ({{ pct(d.ls_adv_pct) }})
  </div>
</div>

<!-- Chart 1: Net Worth Comparison -->
{% if charts|length > 0 %}
<div class="card reveal" style="--d:5">
  <div class="sh">
    <div class="sh-icon i-violet">
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg>
    </div>
    <h2>Net Worth Over Time</h2>
  </div>
  <p class="chart-desc">Median trajectory with 25-75th and 5-95th percentile confidence bands. The shaded regions show the range of likely outcomes across all simulations.</p>
  <img class="chart-img" src="data:image/png;base64,{{ charts[0] }}" alt="Net Worth Comparison">
</div>
{% endif %}

<!-- Chart 2: Outcome Distribution -->
{% if charts|length > 1 %}
<div class="card reveal" style="--d:6">
  <div class="sh">
    <div class="sh-icon i-emerald">
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
    </div>
    <h2>Distribution of Outcomes</h2>
  </div>
  <p class="chart-desc">How final net worth is distributed across all simulations. Wider spread means more uncertainty; the callout shows what percentage of scenarios each strategy wins.</p>
  <img class="chart-img" src="data:image/png;base64,{{ charts[1] }}" alt="Outcome Distribution">
</div>
{% endif %}

<!-- Chart 3: Loan Balance & Investment Pot -->
{% if charts|length > 2 %}
<div class="card reveal" style="--d:7">
  <div class="sh">
    <div class="sh-icon i-indigo">
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7M4 7c0-2 1-3 3-3h10c2 0 3 1 3 3M4 7h16M10 11h4"/></svg>
    </div>
    <h2>Loan Balance &amp; Investment Growth</h2>
  </div>
  <p class="chart-desc">Breaking down the components: how fast each strategy pays down the loan (top) and how investment pots grow over time (bottom).</p>
  <img class="chart-img" src="data:image/png;base64,{{ charts[2] }}" alt="Loan and Investment">
</div>
{% endif %}

<!-- Chart 4: Breakeven Analysis -->
{% if charts|length > 3 %}
<div class="card reveal" style="--d:8">
  <div class="sh">
    <div class="sh-icon i-amber">
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 6h18M3 12h18M3 18h18"/></svg>
    </div>
    <h2>Breakeven Analysis</h2>
  </div>
  <p class="chart-desc">Median net worth for each strategy at different monthly overpayment levels. Shows the threshold where overpaying starts to beat investing.</p>
  <img class="chart-img" src="data:image/png;base64,{{ charts[3] }}" alt="Breakeven Analysis">
</div>
{% endif %}

<!-- Chart 5: Lump Sum Hypothetical -->
{% if charts|length > 4 %}
<div class="card reveal" style="--d:9">
  <div class="sh">
    <div class="sh-icon i-violet">
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z"/></svg>
    </div>
    <h2>Lump Sum Hypothetical: Growth Over Time</h2>
  </div>
  <p class="chart-desc">Deterministic comparison: investing your full loan balance as a lump sum vs paying off the loan and reinvesting the freed mandatory payments. Uses your stated expected return, not Monte Carlo.</p>
  <img class="chart-img" src="data:image/png;base64,{{ charts[4] }}" alt="Lump Sum Hypothetical">
</div>
{% endif %}

<!-- Download PDF -->
<div class="dl-section reveal" style="--d:10">
  <a href="/download-pdf" class="btn btn-success">
    <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
    Download PDF Report
  </a>
</div>

{% endif %}

<div class="footer">Built with Monte Carlo simulation &middot; 10,000+ scenarios per run</div>
</div>

<script>
document.getElementById('sim-form').addEventListener('submit',function(){
  document.getElementById('loading').style.display='flex';
  document.getElementById('submit-btn').disabled=true;
});

/* About toggle */
(function(){
  var btn=document.getElementById('about-toggle');
  var body=document.getElementById('about-body');
  if(!btn||!body) return;
  btn.addEventListener('click',function(){
    var open=btn.getAttribute('aria-expanded')==='true';
    btn.setAttribute('aria-expanded',String(!open));
    body.classList.toggle('open');
  });
})();

/* Staggered reveal on scroll */
(function(){
  var els=document.querySelectorAll('.reveal');
  if(!els.length) return;
  var obs=new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if(e.isIntersecting){
        e.target.classList.add('visible');
        obs.unobserve(e.target);
      }
    });
  },{threshold:0.08,rootMargin:'0px 0px -40px 0px'});
  els.forEach(function(el){obs.observe(el)});
})();
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(
            HTML_TEMPLATE,
            form={},
            d=None,
            be=None,
            charts=[],
            verdict_text="",
            fmt=fmt,
            pct=pct,
        )

    # POST — run simulation
    form = request.form.to_dict()
    inputs = parse_form(form)

    results = run_simulation(inputs)
    be = breakeven_table(inputs)
    sweep = parameter_sweep(inputs, n_iterations=500)
    d = compute_display_data(inputs, results, be)
    verdict_text = generate_verdict_text(d)

    # Generate charts for web display
    chart_images = report.get_web_charts(inputs, results, be, sweep, d)

    # Save PDF for download
    report.generate_pdf(inputs, results, be, sweep, d, verdict_text,
                        PDF_PATH)

    return render_template_string(
        HTML_TEMPLATE,
        form=form,
        d=d,
        be=be,
        charts=chart_images,
        verdict_text=verdict_text,
        fmt=fmt,
        pct=pct,
    )


@app.route("/download-pdf")
def download_pdf():
    if os.path.exists(PDF_PATH):
        return send_file(PDF_PATH, as_attachment=True, download_name="student_loan_report.pdf")
    return "No report generated yet. Run a simulation first.", 404


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

def run_web(debug: bool = True) -> None:
    """Start the Flask development server and open browser."""
    import webbrowser
    import threading

    print("Starting web app at http://localhost:5000")
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=debug)


if __name__ == "__main__":
    run_web()
