"""
UK Tax Constants for Student Loan Plan 2 Simulator.

All monetary values in GBP. Tax year 2025/26 as base year.
Thresholds frozen until April 2028 unless stated otherwise,
then assumed to grow with CPI at 2% per annum.
"""

# ── General assumptions ──────────────────────────────────────────────
CPI_RATE = 0.02          # assumed CPI growth after freeze ends
BASE_TAX_YEAR = 2025     # 2025 means 2025/26

# ── Income Tax (England & Wales) ─────────────────────────────────────
PERSONAL_ALLOWANCE = 12_570
PA_TAPER_THRESHOLD = 100_000       # PA reduces £1 per £2 above this
PA_TAPER_END = 125_140             # PA fully withdrawn here
IT_FREEZE_UNTIL = 2028             # thresholds frozen until April 2028

# Bands: (upper limit, rate). Last band has no upper limit (use inf).
INCOME_TAX_BANDS_ENGLAND = [
    (50_270, 0.20),    # basic rate
    (125_140, 0.40),   # higher rate
    (float("inf"), 0.45),  # additional rate
]

# ── Income Tax (Scotland) ────────────────────────────────────────────
INCOME_TAX_BANDS_SCOTLAND = [
    (15_397, 0.19),    # starter rate
    (27_491, 0.20),    # basic rate
    (43_662, 0.21),    # intermediate rate
    (75_000, 0.42),    # higher rate
    (125_140, 0.45),   # advanced rate
    (float("inf"), 0.48),  # top rate
]

# ── National Insurance (Employee Class 1) ────────────────────────────
NI_FREEZE_UNTIL = 2028
NI_BANDS = [
    (12_570, 0.00),    # below primary threshold
    (50_270, 0.08),    # main rate
    (float("inf"), 0.02),  # upper rate
]

# ── Student Loan Plan 2 ─────────────────────────────────────────────
SL_PLAN2_THRESHOLDS = {
    2025: 28_470,
    2026: 29_385,
    # 2027-2029 frozen at 29,385
}
SL_PLAN2_FREEZE_VALUE = 29_385
SL_PLAN2_FREEZE_UNTIL = 2030       # frozen until April 2030, then RPI
SL_REPAYMENT_RATE = 0.09
SL_INTEREST_UPPER_THRESHOLD = 51_245  # RPI+3% kicks in fully here
SL_WRITE_OFF_YEARS = 30

# ── Investment Tax (GIA) ────────────────────────────────────────────
CGT_ANNUAL_EXEMPT = 3_000
CGT_BASIC_RATE = 0.18
CGT_HIGHER_RATE = 0.24

DIVIDEND_ALLOWANCE = 500
DIVIDEND_BASIC_RATE = 0.0875
DIVIDEND_HIGHER_RATE = 0.3375
DIVIDEND_ADDITIONAL_RATE = 0.3935

ISA_ANNUAL_LIMIT = 20_000

# Portfolio split assumption for GIA
PORTFOLIO_CAPITAL_GROWTH_SHARE = 0.70
PORTFOLIO_DIVIDEND_SHARE = 0.30

# ── Higher-rate threshold (used for investment tax decisions) ────────
BASIC_RATE_LIMIT = 50_270   # taxable income up to this = basic rate
ADDITIONAL_RATE_LIMIT = 125_140
