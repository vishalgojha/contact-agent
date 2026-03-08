import shutil
import uuid
from pathlib import Path

from tools import load_contacts


def _make_test_dir() -> Path:
    path = Path(__file__).resolve().parent / "_runtime" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_load_contacts_from_headerless_csv_keeps_first_phone() -> None:
    test_dir = _make_test_dir()
    try:
        csv_path = test_dir / "numbers.csv"
        csv_path.write_text("+919876543210\n+14155550123\n", encoding="utf-8")

        contacts = load_contacts(str(csv_path), default_region="IN")

        assert [contact.phone for contact in contacts] == ["+919876543210", "+14155550123"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_load_contacts_from_headered_csv_uses_phone_columns() -> None:
    test_dir = _make_test_dir()
    try:
        csv_path = test_dir / "numbers.csv"
        csv_path.write_text("name,phone,notes\nAlice,+919876543210,friend\nBob,+14155550123,work\n", encoding="utf-8")

        contacts = load_contacts(str(csv_path), default_region="IN")

        assert [contact.phone for contact in contacts] == ["+919876543210", "+14155550123"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
