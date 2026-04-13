"""
PII detection and redaction pipeline for tabular data files (CSV / Excel).

This module implements a two-pass redaction strategy:

  Pass 1 — Row-by-row NLP (scrub_text_field):
      Long free-form text columns (e.g. "notes", "comments") are processed one
      cell at a time using Presidio's AnalyzerEngine + AnonymizerEngine.  This
      path is slower but handles arbitrary natural-language sentences where
      entities can span many tokens.

  Pass 2 — Batch columnar scan (scrub_supporting_columns):
      All remaining columns (excluding any the caller explicitly wants to keep
      intact) are processed in bulk via Presidio's StructuredEngine.  This path
      is faster because it analyses whole columns at once instead of calling the
      NLP pipeline per row.

CLI usage example:
    python -m src.replace_pii data/employees.xlsx \\
        --sheet "Sheet1" \\
        --text-fields notes bio \\
        --exclude-fields employee_id \\
        --output data/employees_redacted.xlsx
"""
import argparse

import pandas as pd
from presidio_structured import StructuredEngine, PandasAnalysisBuilder
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine
from faker import Faker

ENTITIES_TO_DETECT = [
    "PERSON", "EMAIL_ADDRESS", "CREDIT_CARD", "CRYPTO",
    "IP_ADDRESS", "MAC_ADDRESS", "PHONE_NUMBER", "MEDICAL_LICENSE",
]

def scrub_text_field(df: pd.DataFrame, field_name: str) -> pd.DataFrame:
    """Redact PII from a single free-form text column using row-by-row NLP analysis.

    This is the slower "Pass 1" path intended for long-form text such as notes
    or comments, where entities may span many words and require sentence-level
    context to be detected accurately.

    Parameters
    ----------
    df : pd.DataFrame
        The source DataFrame.  Modified in place; also returned for chaining.
    field_name : str
        Name of the column to redact.  Every cell in this column is run through
        Presidio's NLP analyzer and then anonymized.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with ``field_name`` values redacted.
    """
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    # Helpful replacement tool
    fake = Faker()

    operators = {
        "PERSON": OperatorConfig("replace", {"new_value": "REDACTED"}),
        "EMAIL_ADDRESS": OperatorConfig("custom", {"lambda": lambda x: fake.safe_email()}),
        "CREDIT_CARD": OperatorConfig("custom", {"lambda": lambda x: fake.credit_card_number()}),
        "CRYPTO": OperatorConfig("replace", {"new_value": "REDACTED"}),
        "IP_ADDRESS": OperatorConfig("replace", {"new_value": "REDACTED"}),
        "MAC_ADDRESS": OperatorConfig("replace", {"new_value": "REDACTED"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "REDACTED"}),
        "MEDICAL_LICENSE": OperatorConfig("replace", {"new_value": "REDACTED"})
    }

    def _anonymize(text):
        # Guard against NaN / None cells: pandas represents missing values as
        # float('nan'), which would cause the analyzer to raise a TypeError.
        if pd.isna(text):
            return text

        text = str(text)
        results = analyzer.analyze(text, language="en", entities=ENTITIES_TO_DETECT)

        return anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text

    print(f"Scrubbing text field '{field_name}'...")

    # Replace the column in the original DataFrame (not a copy) so that the
    # caller sees the changes without needing to reassign the return value.
    df[field_name] = df[field_name].apply(_anonymize)
    return df


def scrub_supporting_columns(df: pd.DataFrame, text_fields: list[str], exclude_fields: list[str],
) -> pd.DataFrame:
    """Redact PII from all columns not already handled by scrub_text_field.

    This is the faster "Pass 2" path that uses Presidio's StructuredEngine to
    analyse and anonymize entire columns at once rather than row-by-row.  It is
    appropriate for short, structured values such as names, phone numbers, and
    email addresses stored in dedicated columns.

    Parameters
    ----------
    df : pd.DataFrame
        The source DataFrame.  Modified in place; also returned for chaining.
    text_fields : list[str]
        Columns already processed by scrub_text_field.  They are excluded here
        to avoid double-processing.
    exclude_fields : list[str]
        Columns that must remain untouched (e.g. primary keys, date columns).

    Returns
    -------
    pd.DataFrame
        The same DataFrame with supporting columns redacted.
    """
    # Build the set of columns to skip.
    skip = set(text_fields) | set(exclude_fields)

    # Collect only the columns that remain after filtering out the skipped ones.
    cols = [col for col in df.columns if col not in skip]

    # Early return if there is nothing left to process.
    if not cols:
        return df

    # StructuredEngine is the batch-mode counterpart to AnalyzerEngine.  It
    # operates on a DataFrame subset rather than on individual strings.
    pandas_engine = StructuredEngine()

    fake = Faker()

    # Same operator mapping as in scrub_text_field; defined here independently
    # so each pass can be configured separately in the future without coupling.
    operators = {
        "PERSON": OperatorConfig("replace", {"new_value": "REDACTED"}),
        "EMAIL_ADDRESS": OperatorConfig("custom", {"lambda": lambda x: fake.safe_email()}),
        "CREDIT_CARD": OperatorConfig("custom", {"lambda": lambda x: fake.credit_card_number()}),
        "CRYPTO": OperatorConfig("replace", {"new_value": "REDACTED"}),
        "IP_ADDRESS": OperatorConfig("replace", {"new_value": "REDACTED"}),
        "MAC_ADDRESS": OperatorConfig("replace", {"new_value": "REDACTED"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "REDACTED"}),
        "MEDICAL_LICENSE": OperatorConfig("replace", {"new_value": "REDACTED"})
    }

    print("Scrubbing supporting columns...")
    subset = df[cols].copy()

    # Presidio's StructuredEngine requires object (string) dtype.  Cast every
    # column to str so numeric/date/bool columns are analysed, not silently
    # skipped.  Capture the NaN mask first so that NaN cells are restored
    # after conversion — str(NaN) would otherwise produce the literal "nan".
    na_mask = subset.isna()
    # astype(str) on nullable extension types (e.g. Int64) produces StringDtype,
    # not object.  The second astype(object) normalises every column to plain
    # object dtype, which is what Presidio's StructuredEngine requires.
    subset = subset.astype(str).astype(object)
    subset[na_mask] = pd.NA

    analysis = PandasAnalysisBuilder().generate_analysis(subset, entity_types=ENTITIES_TO_DETECT)
    anonymized = pandas_engine.anonymize(subset, analysis, operators=operators)

    # Write back only the named columns rather than replacing `df` wholesale.
    df[cols] = anonymized[cols]
    return df


def main():
    """CLI entry point: parse arguments, load the file, and run both redaction passes."""
    parser = argparse.ArgumentParser(description="Detect and redact PII from a data file")

    # Positional: the file to redact.  Supports .csv, .xls, and .xlsx.
    parser.add_argument("filepath", help="Path to the input file (.xlsx, .xls, or .csv)")

    # Optional: which Excel sheet to read.  Defaults to 0 (the first sheet by
    # index).  Accepts either an integer index or a sheet name string.
    parser.add_argument("--sheet", default=0,
                        help="Sheet name or index for Excel files (default: 0)")

    # Optional: one or more column names that contain free-form text and should
    # be processed with the slower row-by-row NLP path (Pass 1).
    parser.add_argument("--text-fields", nargs="+", default=[], metavar="FIELD",
                        help="Long-form text columns to scrub row-by-row")

    # Optional: one or more column names to leave completely untouched (e.g.
    # primary keys or reference IDs that must remain unchanged for traceability).
    parser.add_argument("--exclude-fields", nargs="+", default=[], metavar="FIELD",
                        help="Columns to leave untouched")

    parser.add_argument(
        "--output", required=True,
        help="Path for the redacted output file (.csv, .xls, or .xlsx)"
    )

    args = parser.parse_args()
    if args.filepath.lower().endswith(".csv"):
        try:
            df = pd.read_csv(args.filepath, encoding='utf-8-sig').fillna("")
        except:
            df = pd.read_csv(args.filepath, encoding='latin1').fillna("")

    else:
        df = pd.read_excel(args.filepath, sheet_name=args.sheet).fillna("")


    # Normalise column names: strip leading/trailing whitespace so that headers
    # with incidental spaces (" first_name ") match user-supplied field names.
    df.columns = df.columns.str.strip()

    # Pass 1 — row-by-row NLP redaction for free-form text columns.
    for field in args.text_fields:
        # Silently skip any field name that does not exist in this file.
        if field in df.columns:
            df = scrub_text_field(df, field)

    # Pass 2 — batch columnar redaction for all remaining columns.
    # text_fields is passed again so that scrub_supporting_columns knows to
    # exclude those columns even though they were already processed
    df = scrub_supporting_columns(df, args.text_fields, args.exclude_fields)

    out = args.output.lower()
    if out.endswith(".csv"):
        df.to_csv(args.output, index=False)
    elif out.endswith(".xlsx") or out.endswith(".xls"):
        df.to_excel(args.output, index=False)
    else:
        parser.error(f"Unsupported output format: {args.output}. Use .csv, .xls, or .xlsx.")

    print(f"Redacted file written to {args.output}")
    return df


if __name__ == "__main__":
    main()
