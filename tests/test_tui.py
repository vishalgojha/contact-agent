import shutil
import uuid
from pathlib import Path

from memory import ContactRecord
from tui import discover_google_contact_candidates, discover_input_candidates, summarize_records


def _make_test_dir() -> Path:
    path = Path(__file__).resolve().parent / "_runtime" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_discover_input_candidates_excludes_google_contacts_exports() -> None:
    test_dir = _make_test_dir()
    try:
        (test_dir / "numbers.csv").write_text("phone\n+919876543210\n", encoding="utf-8")
        (test_dir / "google_contacts.csv").write_text("Name,Phone 1 - Value\n", encoding="utf-8")
        (test_dir / "notes.md").write_text("ignore", encoding="utf-8")

        candidates = discover_input_candidates(test_dir)

        assert [path.name for path in candidates] == ["numbers.csv"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_discover_google_contact_candidates_only_returns_contact_exports() -> None:
    test_dir = _make_test_dir()
    try:
        (test_dir / "numbers.csv").write_text("phone\n+919876543210\n", encoding="utf-8")
        (test_dir / "my_contacts.vcf").write_text("BEGIN:VCARD\nEND:VCARD\n", encoding="utf-8")
        (test_dir / "google_contacts.xlsx").write_text("placeholder", encoding="utf-8")

        candidates = discover_google_contact_candidates(test_dir)

        assert [path.name for path in candidates] == ["google_contacts.xlsx", "my_contacts.vcf"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_summarize_records_counts_sources_and_unknowns() -> None:
    records = [
        ContactRecord(
            phone="+919876543210",
            name="Alice",
            confidence=0.99,
            source="google_contacts",
            reasoning="Direct match.",
            last_updated="2026-03-08T00:00:00Z",
        ),
        ContactRecord(
            phone="+14155550123",
            name="UNKNOWN",
            confidence=0.0,
            source="unknown",
            reasoning="No evidence.",
            last_updated="2026-03-08T00:00:00Z",
        ),
        ContactRecord(
            phone="+14155550124",
            name="Bob",
            confidence=0.88,
            source="truecaller_cli",
            reasoning="Local Truecaller match.",
            last_updated="2026-03-08T00:00:00Z",
        ),
    ]

    summary = summarize_records(records)

    assert summary["total"] == 3
    assert summary["resolved"] == 2
    assert summary["unknown"] == 1
    assert summary["source_counts"] == {"google_contacts": 1, "truecaller_cli": 1, "unknown": 1}
