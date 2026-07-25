"""
Parsing for the AI Ecosystem reference files (config/reference/layers/*.md).

Pure/pandas-only logic, shared between dashboard/app.py's layers_tab()
(display) and buffett/scanner_ecosystem.py (scanning) -- previously this
parsing lived only inside dashboard/app.py, which would have meant
duplicating ~170 lines of markdown-table-parsing logic to reuse it from a
scanner, the same drift risk that caused buffett/scanner_ai.py and
buffett/scanner_etf.py to fall out of sync with the main scanner.
"""
import re
from typing import List

import pandas as pd

LAYER_FILES = {
    "⚡ Layer 1 - Energy": "config/reference/layers/Layer 1  Energy Companies Powering the AI Infrastructure Buildout.md",
    "💻 Layer 2 - Chips & Computers": "config/reference/layers/Layer 2  Chips and Computers – Listed US and Hong Kong China Companies Powering AI Compute.md",
    "🏢 Layer 3 - Infrastructure": "config/reference/layers/Layer 3  AI Infrastructure – Data Centers, Land, and Power-Adjacent Real Assets.md",
    "🧠 Layer 4 - AI Models": "config/reference/layers/Layer 4  Model Layer – Listed US and Hong Kong China AI Model and Platform Companies.md",
    "🚀 Layer 5 - Applications": "config/reference/layers/Layer 5  Application Layer – Listed US and Hong Kong China AI-Enabled Companies.md",
}

# Canonical names of ticker columns across the five layer files
TICKER_CANDIDATES = ['Ticker', 'Ticker (US)', 'Ticker / Listing', 'Listing',
                     'Ticker(s)', 'Symbol', 'Ticker (HKEX)']
COMPANY_CANDIDATES = ['Company', 'Company / Group']
REGION_CANDIDATES = ['Region']

# Map ticker suffix → region for non-US exchanges (US is the default when no suffix)
_REGION_BY_SUFFIX = [
    (re.compile(r'\.HK$'), 'HK'),
    (re.compile(r'\.SH$'), 'China'),
    (re.compile(r'\.SS$'), 'China'),
    (re.compile(r'\.SZ$'), 'China'),
    (re.compile(r'\.T$'), 'Japan'),
    (re.compile(r'\.L$'), 'UK ADR'),
    (re.compile(r'\.LON$'), 'UK ADR'),
    (re.compile(r'\.TO$'), 'Canada'),
    (re.compile(r'\.DE$'), 'Germany ADR'),
]


def parse_layer_markdown(file_path):
    """Extract ALL markdown tables from a layer file as a list of DataFrames.

    Returns a list of (section_name, DataFrame) tuples — section name is the
    most recent heading above the table, useful as a sub-layer label.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Walk line-by-line, track current heading, accumulate table rows.
    tables = []
    cur_section = file_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    cur = []
    in_table = False
    for line in lines:
        s = line.rstrip('\n')
        stripped = s.strip()
        # a markdown table row starts AND ends with a pipe, with >=2 pipes
        if stripped.startswith('|') and stripped.endswith('|') and stripped.count('|') >= 3:
            in_table = True
            cur.append(s)
        else:
            if in_table:
                # heading above the table — promote it as section name
                if stripped.startswith('#') or stripped.startswith('**'):
                    heading = stripped.lstrip('#').strip().strip('*').strip()
                    if heading:
                        cur_section = heading
            in_table = False
            if len(cur) >= 3:
                tables.append((cur_section, _md_table_to_df(cur)))
            cur = []
            # capture standalone headings for next-table section
            if stripped.startswith('#'):
                cur_section = stripped.lstrip('#').strip().strip('*').strip()
    if len(cur) >= 3:
        tables.append((cur_section, _md_table_to_df(cur)))
    return tables


def _md_table_to_df(raw_lines):
    """Convert raw markdown table lines to a DataFrame."""
    cleaned = [re.sub(r'^\s*\|\s*|\s*\|\s*$', '', l) for l in raw_lines]
    data = [[c.strip() for c in l.split('|')] for l in cleaned]
    if len(data) < 2:
        return pd.DataFrame()
    header = data[0]
    # find separator row (---)
    sep_idx = None
    for i, row in enumerate(data[1:], 1):
        if all(re.match(r'^[\s\-:]+$', c) for c in row) and len(row) == len(header):
            sep_idx = i
            break
    rows = data[sep_idx+1:] if sep_idx is not None else data[1:]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=header)
    # Strip whitespace-only fake columns (caused by leading/trailing pipes)
    df = df.loc[:, ~df.columns.str.match(r'^\s*$')]
    return df


def _parse_ticker_token(s):
    """Pull a single clean ticker token from a string like 'NVDA (NASDAQ)' or '0992.HK'."""
    s = str(s).strip()
    m = re.match(r'^([0-9]{4,6}\.[A-Z]{1,3}|[A-Z]{1,5}(?:\.[A-Z])?)\b', s)
    return m.group(1) if m else None


def _split_tickers(cell):
    """Split a multi-ticker cell like '0992.HK (HKEX), LNVGY (ADR)' into a list of clean tickers."""
    s = str(cell)
    parts = re.split(r'[,/]', s)
    out = []
    for p in parts:
        t = _parse_ticker_token(p)
        if t and t not in out:
            out.append(t)
    return out


def _region_from_ticker(ticker):
    """Infer region from ticker suffix. Default to 'US'."""
    for rx, region in _REGION_BY_SUFFIX:
        if rx.search(ticker):
            return region
    return 'US'


def enrich_ticker_rows(combined_df):
    """For each row, find the ticker source column with a non-empty value.

    pd.concat() of differently-shaped tables leaves a 'Ticker' key on every
    row (NaN for rows whose source table doesn't have that column), so a
    pure column-name lookup is misleading. We must look per-row.
    """
    # Determine which of the candidate columns are actually present in this df
    tk_candidates = [c for c in TICKER_CANDIDATES if c in combined_df.columns]

    has_region = 'Region' in combined_df.columns

    tks, all_tks, regions = [], [], []
    for _, row in combined_df.iterrows():
        # Pick the first ticker source where THIS row has a non-empty value
        cell = ''
        for cand in tk_candidates:
            val = row.get(cand)
            if pd.notna(val) and str(val).strip():
                cell = str(val).strip()
                break

        tokens = _split_tickers(cell) if cell else []
        primary = tokens[0] if tokens else ''

        tks.append(primary)
        all_tks.append('|'.join(tokens))

        # Region inference — try the row's own Region value first; then ticker suffix
        reg = '-'
        if has_region:
            raw = row.get('Region')
            if pd.notna(raw):
                reg = str(raw).strip()
        if not reg or reg == '-':
            if primary:
                reg = _region_from_ticker(primary)
            else:
                reg = '-'
        else:
            # Normalize common variants
            if 'US' in reg:
                reg = 'US'
            elif 'HK' in reg:
                reg = 'HK'
            elif 'China' in reg:
                reg = 'China'
            else:
                reg = _region_from_ticker(primary) if primary else reg

        regions.append(reg)

    combined_df['Ticker'] = tks
    combined_df['AllTickers'] = all_tks
    combined_df['Region'] = regions
    return combined_df


def load_all_layers_combined(layer_files=None) -> pd.DataFrame:
    """Parse every layer file and return one combined, ticker-enriched
    DataFrame (Ticker/Region/Layer/SubLayer columns, one row per company)."""
    layer_files = layer_files or LAYER_FILES
    rows = []
    for layer_name, file_path in layer_files.items():
        try:
            tables = parse_layer_markdown(file_path)
        except Exception:
            continue
        for sub_label, df in tables:
            if df.empty:
                continue
            df = df.copy()
            df['Layer'] = layer_name
            df['SubLayer'] = sub_label
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    return enrich_ticker_rows(combined)


def get_all_ecosystem_tickers(layer_files=None) -> List[str]:
    """Return the deduplicated list of primary tickers across all layer
    files, in first-seen order. Used both to display coverage and to scan
    the full AI Ecosystem reference through the main scoring pipeline
    (buffett/scanner_ecosystem.py)."""
    combined = load_all_layers_combined(layer_files)
    if combined.empty or 'Ticker' not in combined.columns:
        return []
    seen = set()
    tickers = []
    for t in combined['Ticker']:
        t = str(t).strip()
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)
    return tickers
