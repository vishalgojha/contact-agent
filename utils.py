"""Shared utilities for logging, timestamps, and phone handling."""
# contact_name_agent/utils.py

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import datetime, timezone

import phonenumbers
from phonenumbers import PhoneNumberFormat

LOGGER_NAME = "contact_name_agent"
PHONE_COLUMN_HINTS = {
    "phone",
    "phone_number",
    "phone number",
    "number",
    "mobile",
    "mobile_number",
    "mobile number",
    "telephone",
    "tel",
    "contact",
}


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the shared application logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def mask_phone(phone: str) -> str:
    """Mask a phone number for logs and session history."""
    digits = re.sub(r"\D", "", phone or "")
    if not digits:
        return "unknown"
    if len(digits) <= 4:
        return f"***{digits}"
    prefix = "+" if str(phone).startswith("+") else ""
    return f"{prefix}***{digits[-4:]}"


def normalize_phone(value: str, default_region: str = "IN") -> str | None:
    """Normalize an arbitrary phone number into E.164 format."""
    raw = str(value or "").strip()
    if not raw:
        return None

    region = None if raw.startswith("+") else default_region.upper()
    try:
        parsed = phonenumbers.parse(raw, region)
    except phonenumbers.NumberParseException:
        return None

    if not phonenumbers.is_possible_number(parsed):
        return None
    if not phonenumbers.is_valid_number(parsed):
        return None

    return phonenumbers.format_number(parsed, PhoneNumberFormat.E164)


def normalize_region_input(value: str) -> str | None:
    """Normalize an ISO region or dialing code string into an ISO region code."""
    raw = str(value or "").strip().upper()
    if not raw:
        return None

    if raw.startswith("+"):
        raw = raw[1:]

    if raw.isdigit():
        try:
            region = phonenumbers.region_code_for_country_code(int(raw))
        except ValueError:
            return None
        if region and region != "001":
            return region.upper()
        return None

    if raw in phonenumbers.SUPPORTED_REGIONS:
        return raw
    return None


def country_code_for_region(region: str) -> int | None:
    """Return the country calling code for an ISO region."""
    normalized = normalize_region_input(region)
    if normalized is None:
        return None
    code = phonenumbers.country_code_for_region(normalized)
    return int(code) if code else None


def extract_phone_candidates(text: str) -> list[str]:
    """Extract phone-like substrings from a free-form text blob."""
    pattern = re.compile(r"(?:\+|00)?\d[\d\-\s\(\)]{6,}\d")
    matches = pattern.findall(text or "")
    return dedupe_strings(match.strip() for match in matches)


def dedupe_strings(values: Iterable[str]) -> list[str]:
    """Deduplicate strings while preserving order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def clamp_confidence(value: float | int) -> float:
    """Clamp a numeric confidence value into the inclusive 0-1 range."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))
