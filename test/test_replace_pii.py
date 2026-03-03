import sys
import pandas as pd
import pytest
from unittest.mock import patch

from src.replace_pii import main


def test_header_spaces_stripped(tmp_path, monkeypatch):
    """Column headers with leading/trailing spaces must be stripped after loading."""
    csv_in = tmp_path / "input.csv"
    csv_out = tmp_path / "output.csv"

    # Write a CSV whose headers have deliberate surrounding spaces.
    csv_in.write_text(" first_name , email \nAlice,alice@example.com\n")

    monkeypatch.setattr(
        sys, "argv",
        ["replace_pii", str(csv_in), "--output", str(csv_out)],
    )

    # Bypass the NLP pipeline so the test is fast and dependency-free.
    with patch("src.replace_pii.scrub_text_field", side_effect=lambda df, f: df), \
         patch("src.replace_pii.scrub_supporting_columns", side_effect=lambda df, tf, ef: df):
        main()

    result = pd.read_csv(csv_out)
    assert list(result.columns) == ["first_name", "email"]


def test_numeric_columns_cast_to_string(monkeypatch):
    """Numeric and NaN columns must be cast to str without producing 'nan' strings."""
    import pandas as pd
    from unittest.mock import patch, MagicMock

    df = pd.DataFrame({
        "phone": pd.array([5551234567, pd.NA], dtype="Int64"),
        "score": [9.5, None],
    })

    captured = {}

    def fake_generate_analysis(subset_df):
        captured["subset"] = subset_df.copy()
        return MagicMock()

    def fake_anonymize(subset_df, analysis, operators):
        # Identity: return the subset unchanged so we can check write-back.
        return subset_df

    with patch("src.replace_pii.PandasAnalysisBuilder") as MockBuilder, \
         patch.object(__import__("src.replace_pii", fromlist=["StructuredEngine"]).StructuredEngine,
                      "anonymize", side_effect=fake_anonymize):
        MockBuilder.return_value.generate_analysis.side_effect = fake_generate_analysis
        from src.replace_pii import scrub_supporting_columns
        result = scrub_supporting_columns(df.copy(), text_fields=[], exclude_fields=[])

    subset = captured["subset"]
    # All values sent to Presidio must be str (object dtype), not int/float.
    assert subset["phone"].dtype == object
    assert subset["score"].dtype == object
    # NaN cells must remain NaN — not the literal string "nan".
    assert pd.isna(subset["phone"].iloc[1])
    assert pd.isna(subset["score"].iloc[1])
