"""
Line items of the financial statements used by Step 2 (src/modeling/step2_funds.py).

The cached and downloaded statements (src/ingestion/funds_api.py,
scripts/update_fundamentals.py) are filtered to these rows.
"""
# ==========================================
# FUNDAMENTAL UNIVERSE (Step 2)
# ==========================================
INCOME_STATEMENT = [
    'Total Revenue', 'Operating Revenue', 'Gross Profit', 'Operating Income',
    'EBIT', 'EBITDA', 'Net Income', 'Net Income Common Stockholders',
    'Basic EPS', 'Diluted EPS'
]

BALANCE_SHEET = [
    'Total Assets', 'Current Assets', 'Total Liabilities Net Minority Interest',
    'Current Liabilities', 'Total Debt', 'Net Debt', 'Cash And Cash Equivalents',
    'Stockholders Equity', 'Working Capital', 'Retained Earnings'
]

CASH_FLOW = [
    'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow',
    'Free Cash Flow', 'Capital Expenditure', 'Repayment Of Debt', 'Issuance Of Debt'
]

FUNDAMENTAL_UNIVERSE = INCOME_STATEMENT + BALANCE_SHEET + CASH_FLOW
