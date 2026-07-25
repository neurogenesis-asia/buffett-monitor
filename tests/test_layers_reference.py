"""
Real assertions for buffett/layers_reference.py -- the markdown-table
parsing shared between dashboard/app.py's layers_tab() (display) and
buffett/scanner_ecosystem.py (scanning). Previously this logic lived
only in dashboard/app.py; these tests exercise the extracted module
directly against synthetic markdown fixtures (no dependency on the
real config/reference/layers/*.md content, which can change).
"""
import pytest

from buffett.layers_reference import (
    parse_layer_markdown,
    enrich_ticker_rows,
    get_all_ecosystem_tickers,
    load_all_layers_combined,
    _parse_ticker_token,
    _split_tickers,
    _region_from_ticker,
)


SAMPLE_MD = """# Layer 1 - Energy

## Utilities

| Company | Ticker | Notes |
|---------|--------|-------|
| NextEra Energy | NEE | Largest US utility |
| Constellation Energy | CEG | Nuclear exposure |

## HK Names

| Company | Ticker | Region |
|---------|--------|--------|
| CLP Holdings | 0002.HK | HK |
"""


def test_parse_layer_markdown_extracts_multiple_tables(tmp_path):
    md_path = tmp_path / "layer1.md"
    md_path.write_text(SAMPLE_MD)

    tables = parse_layer_markdown(str(md_path))

    assert len(tables) == 2
    section_names = [name for name, _ in tables]
    assert "Utilities" in section_names
    assert "HK Names" in section_names


def test_parse_layer_markdown_table_content_is_correct(tmp_path):
    md_path = tmp_path / "layer1.md"
    md_path.write_text(SAMPLE_MD)

    tables = parse_layer_markdown(str(md_path))
    utilities_df = next(df for name, df in tables if name == "Utilities")

    assert list(utilities_df["Ticker"]) == ["NEE", "CEG"]
    assert list(utilities_df["Company"]) == ["NextEra Energy", "Constellation Energy"]


def test_parse_layer_markdown_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_layer_markdown(str(tmp_path / "nope.md"))


# ---------------------------------------------------------------------------
# ticker token parsing
# ---------------------------------------------------------------------------

def test_parse_ticker_token_plain_us_ticker():
    assert _parse_ticker_token("NVDA") == "NVDA"


def test_parse_ticker_token_with_exchange_suffix():
    assert _parse_ticker_token("0992.HK") == "0992.HK"


def test_parse_ticker_token_with_trailing_annotation():
    assert _parse_ticker_token("NVDA (NASDAQ)") == "NVDA"


def test_split_tickers_multi_value_cell():
    assert _split_tickers("0992.HK (HKEX), LNVGY (ADR)") == ["0992.HK", "LNVGY"]


def test_region_from_ticker_suffix():
    assert _region_from_ticker("0992.HK") == "HK"
    assert _region_from_ticker("3800.HK") == "HK"
    assert _region_from_ticker("NVDA") == "US"


# ---------------------------------------------------------------------------
# enrich_ticker_rows / load_all_layers_combined / get_all_ecosystem_tickers
# ---------------------------------------------------------------------------

def test_enrich_ticker_rows_infers_region_from_suffix_when_no_region_column():
    import pandas as pd
    df = pd.DataFrame({"Ticker": ["NVDA", "0992.HK"], "Company": ["Nvidia", "CLP"]})
    enriched = enrich_ticker_rows(df)
    assert list(enriched["Region"]) == ["US", "HK"]


def test_enrich_ticker_rows_uses_explicit_region_column_when_present():
    import pandas as pd
    df = pd.DataFrame({
        "Ticker": ["NVDA"], "Company": ["Nvidia"], "Region": ["United States"],
    })
    enriched = enrich_ticker_rows(df)
    assert enriched["Region"].iloc[0] == "US"


def test_load_all_layers_combined_merges_all_files(tmp_path):
    layer1 = tmp_path / "layer1.md"
    layer1.write_text(SAMPLE_MD)
    layer2_md = "# Layer 2\n\n| Company | Ticker |\n|---|---|\n| Nvidia | NVDA |\n"
    layer2 = tmp_path / "layer2.md"
    layer2.write_text(layer2_md)

    combined = load_all_layers_combined({"Layer 1": str(layer1), "Layer 2": str(layer2)})

    assert not combined.empty
    assert set(combined["Ticker"]) >= {"NEE", "CEG", "0002.HK", "NVDA"}
    assert set(combined["Layer"]) == {"Layer 1", "Layer 2"}


def test_get_all_ecosystem_tickers_deduplicates_across_files(tmp_path):
    layer1 = tmp_path / "layer1.md"
    layer1.write_text("# L1\n\n| Company | Ticker |\n|---|---|\n| Nvidia | NVDA |\n")
    layer2 = tmp_path / "layer2.md"
    layer2.write_text("# L2\n\n| Company | Ticker |\n|---|---|\n| Nvidia | NVDA |\n| AMD | AMD |\n")

    tickers = get_all_ecosystem_tickers({"Layer 1": str(layer1), "Layer 2": str(layer2)})

    assert tickers == ["NVDA", "AMD"]


def test_get_all_ecosystem_tickers_empty_when_no_files_resolve(tmp_path):
    tickers = get_all_ecosystem_tickers({"Bad": str(tmp_path / "nope.md")})
    assert tickers == []
