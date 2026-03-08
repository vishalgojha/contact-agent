"""Rich-based onboarding and summary helpers for the Contact Name Agent."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from memory import ContactRecord, MemoryStore
from utils import mask_phone, normalize_region_input

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:  # pragma: no cover - optional dependency at runtime
    Console = None
    Panel = None
    Table = None
    Text = None
    box = None


ACCENT = "#ff6a00"
MUTED = "grey66"
SUCCESS = "green"
WARNING = "yellow"
SUPPORTED_INPUT_SUFFIXES = {".csv", ".xlsx", ".txt", ".json", ".vcf"}
GOOGLE_CONTACT_SUFFIXES = {".csv", ".xlsx", ".vcf"}


@dataclass(frozen=True)
class OnboardingConfig:
    """Collected settings from the terminal onboarding flow."""

    input_path: Path
    google_contacts_path: Path | None
    default_region: str
    model: str
    allow_web: bool
    max_iterations: int
    ollama_host: str
    allow_truecaller: bool


def discover_input_candidates(search_root: Path) -> list[Path]:
    """Return likely number-input files from the local inputs directory."""
    return [
        path.resolve()
        for path in _discover_files(search_root, allowed_suffixes=SUPPORTED_INPUT_SUFFIXES)
        if not _looks_like_google_contacts_filename(path.name)
    ]


def discover_google_contact_candidates(search_root: Path) -> list[Path]:
    """Return likely Google Contacts exports from the local inputs directory."""
    return [
        path.resolve()
        for path in _discover_files(search_root, allowed_suffixes=GOOGLE_CONTACT_SUFFIXES)
        if _looks_like_google_contacts_filename(path.name)
    ]


def summarize_records(records: Sequence[ContactRecord]) -> dict[str, object]:
    """Build a compact run summary for TUI display."""
    total = len(records)
    resolved = sum(1 for record in records if record.name != "UNKNOWN")
    source_counts = Counter(record.source for record in records)
    return {
        "total": total,
        "resolved": resolved,
        "unknown": total - resolved,
        "source_counts": dict(sorted(source_counts.items())),
    }


class ContactAgentTUI:
    """Small, dependency-light TUI for onboarding and completion summaries."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.console = Console() if Console is not None else None

    def run_onboarding(self, *, memory_store: MemoryStore, truecaller_ready: bool, truecaller_reason: str) -> OnboardingConfig | None:
        """Collect a runnable agent configuration interactively."""
        saved_region = memory_store.get_default_region() or "IN"
        input_candidates = discover_input_candidates(self.project_root / "inputs")
        google_candidates = discover_google_contact_candidates(self.project_root / "inputs")

        self.print_heading("Contact Name Agent onboarding")
        self.print_panel(
            "Privacy",
            [
                "This agent stays local by default.",
                "Public web search is opt-in and disabled unless you enable it.",
                "Resolved contacts are stored locally in memory/resolved_contacts.json.",
                "Masked session history is stored in memory/session_history.json.",
                "Unknown is acceptable when the evidence is weak.",
            ],
        )
        if not self.ask_yes_no("I understand the local-first privacy model. Continue?", default=True):
            self.print_note("Onboarding cancelled.")
            return None

        mode = self.ask_choice(
            "Onboarding mode",
            [
                ("quickstart", "QuickStart", "Use safe defaults and ask only for the essentials."),
                ("custom", "Custom", "Tune model, host, and iteration limits before running."),
            ],
            default_key="quickstart",
        )

        if input_candidates:
            self.print_panel("Detected input files", [self._display_path(path) for path in input_candidates])
        input_path = self.ask_existing_path(
            "Input path",
            default=input_candidates[0] if input_candidates else self.project_root / "inputs" / "numbers.csv",
            allow_blank=False,
            error_message="Enter an existing file or directory containing phone numbers.",
        )

        if google_candidates:
            self.print_panel(
                "Detected Google Contacts files (optional)",
                [self._display_path(path) for path in google_candidates],
            )
        google_default = google_candidates[0] if google_candidates else None
        google_contacts_path = self.ask_existing_path(
            "Google Contacts path (optional)",
            default=google_default,
            allow_blank=True,
            error_message="Enter an existing Google Contacts export path or leave blank to skip.",
            blank_means_none=True,
        )

        default_region = self.ask_text(
            "Default region",
            default=saved_region,
            validator=lambda value: normalize_region_input(value) is not None,
            error_message="Use an ISO country like IN or US, or a dialing code like +91.",
        ).upper()

        model = "llama3.2:latest"
        ollama_host = "http://127.0.0.1:11434"
        max_iterations = 4

        if mode == "custom":
            model_mode = self.ask_choice(
                "Ollama model",
                [
                    ("default", "Keep recommended model", "Use llama3.2:latest."),
                    ("custom", "Enter model manually", "Provide any local Ollama model tag."),
                ],
                default_key="default",
            )
            if model_mode == "custom":
                model = self.ask_text("Model name", default=model, validator=lambda value: bool(value.strip()))
            ollama_host = self.ask_text(
                "Ollama host",
                default=ollama_host,
                validator=lambda value: bool(value.strip()),
            )
            max_iterations = int(
                self.ask_text(
                    "Max iterations per contact",
                    default=str(max_iterations),
                    validator=lambda value: value.isdigit() and int(value) > 0,
                    error_message="Enter a whole number greater than zero.",
                )
            )
        else:
            self.print_panel(
                "Run defaults",
                [
                    f"Model: {model}",
                    f"Ollama host: {ollama_host}",
                    f"Max iterations: {max_iterations}",
                ],
            )

        allow_web = self.ask_yes_no(
            "Allow public web search as a fallback?",
            default=False,
            note="Recommended default: no. This keeps the run fully local.",
        )

        if truecaller_ready:
            allow_truecaller = self.ask_yes_no(
                "Enable Truecaller CLI lookups?",
                default=True,
                note="Truecaller is configured locally and can be used for Indian mobile numbers.",
            )
        else:
            allow_truecaller = False
            self.print_panel(
                "Truecaller status",
                [
                    "Truecaller is disabled for this run.",
                    truecaller_reason or "The local truecaller-cli setup is incomplete.",
                    "You can configure it later with `python agent.py setup-truecaller --number YOUR_10_DIGIT_NUMBER`.",
                ],
            )

        config = OnboardingConfig(
            input_path=input_path,
            google_contacts_path=google_contacts_path,
            default_region=normalize_region_input(default_region) or saved_region,
            model=model,
            allow_web=allow_web,
            max_iterations=max_iterations,
            ollama_host=ollama_host,
            allow_truecaller=allow_truecaller,
        )
        self.show_review(config)
        if not self.ask_yes_no("Start recovery now?", default=True):
            self.print_note("Run cancelled.")
            return None
        return config

    def show_review(self, config: OnboardingConfig) -> None:
        """Display the resolved onboarding configuration before launch."""
        rows = [
            ("Input", self._display_path(config.input_path)),
            ("Google contacts", self._display_path(config.google_contacts_path) if config.google_contacts_path else "Skipped"),
            ("Default region", config.default_region),
            ("Model", config.model),
            ("Ollama host", config.ollama_host),
            ("Web search", "Enabled" if config.allow_web else "Disabled"),
            ("Truecaller", "Enabled" if config.allow_truecaller else "Disabled"),
            ("Max iterations", str(config.max_iterations)),
        ]
        self.print_key_value_panel("Review", rows)

    def show_run_summary(self, *, records: Sequence[ContactRecord], export_path: Path | None) -> None:
        """Render a concise end-of-run summary."""
        summary = summarize_records(records)
        rows = [
            ("Total contacts", str(summary["total"])),
            ("Resolved", str(summary["resolved"])),
            ("Unknown", str(summary["unknown"])),
            ("Export", self._display_path(export_path) if export_path else "Not written"),
        ]
        self.print_key_value_panel("Run summary", rows)

        source_counts = summary["source_counts"]
        if source_counts:
            self.print_key_value_panel(
                "Resolution sources",
                [(source, str(count)) for source, count in source_counts.items()],
            )

        if records:
            self.print_results_table(records[:10])

    def print_results_table(self, records: Sequence[ContactRecord]) -> None:
        """Show the first few resolved rows in a compact table."""
        if self.console is None or Table is None:
            print("Preview:")
            for record in records:
                print(f"- {mask_phone(record.phone)} | {record.name} | {record.source} | {record.confidence:.2f}")
            return

        table = Table(title="Preview", box=box.SIMPLE_HEAVY, show_lines=False)
        table.add_column("Phone", style="white")
        table.add_column("Name", style="white")
        table.add_column("Source", style=MUTED)
        table.add_column("Confidence", justify="right", style=SUCCESS)
        for record in records:
            table.add_row(mask_phone(record.phone), record.name, record.source, f"{record.confidence:.2f}")
        self.console.print(table)

    def print_heading(self, title: str) -> None:
        """Render a top-level heading."""
        if self.console is None:
            print(title)
            return
        self.console.print(Text(title, style=f"bold {ACCENT}"))

    def print_panel(self, title: str, lines: Sequence[str]) -> None:
        """Render a titled note panel."""
        content = "\n".join(lines)
        if self.console is None or Panel is None:
            print(f"\n{title}\n{content}\n")
            return
        self.console.print(
            Panel.fit(
                content,
                title=f"[bold {ACCENT}]{title}[/]",
                border_style=MUTED,
                box=box.SQUARE,
            )
        )

    def print_key_value_panel(self, title: str, rows: Sequence[tuple[str, str]]) -> None:
        """Render a review panel with left-right key/value rows."""
        if self.console is None or Table is None or Panel is None:
            print(f"\n{title}")
            for key, value in rows:
                print(f"{key}: {value}")
            return

        table = Table.grid(padding=(0, 2))
        table.add_column(style=f"bold {ACCENT}", justify="left")
        table.add_column(style="white", justify="left")
        for key, value in rows:
            table.add_row(key, value)
        self.console.print(
            Panel.fit(table, title=f"[bold {ACCENT}]{title}[/]", border_style=MUTED, box=box.SQUARE)
        )

    def print_note(self, message: str) -> None:
        """Render a simple status note."""
        if self.console is None:
            print(message)
            return
        self.console.print(Text(message, style=MUTED))

    def ask_choice(
        self,
        title: str,
        options: Sequence[tuple[str, str, str]],
        *,
        default_key: str,
    ) -> str:
        """Prompt the user to choose one option by number."""
        default_index = next((index for index, option in enumerate(options, start=1) if option[0] == default_key), 1)
        if self.console is None:
            print(f"\n{title}")
            for index, (_, label, description) in enumerate(options, start=1):
                print(f"{index}. {label} - {description}")
        else:
            self.console.print(Text(title, style=f"bold {ACCENT}"))
            for index, (_, label, description) in enumerate(options, start=1):
                marker = "*" if index == default_index else "-"
                self.console.print(f"  {marker} {index}. {label} [{MUTED}]{description}[/]")

        while True:
            raw = input(f"Select option [{default_index}]: ").strip()
            if not raw:
                return options[default_index - 1][0]
            if raw.isdigit():
                index = int(raw)
                if 1 <= index <= len(options):
                    return options[index - 1][0]
            self.print_note("Enter one of the listed option numbers.")

    def ask_yes_no(self, prompt: str, *, default: bool, note: str = "") -> bool:
        """Prompt for a yes/no answer."""
        if self.console is not None:
            self.console.print(Text(prompt, style=f"bold {ACCENT}"))
            if note:
                self.console.print(Text(note, style=MUTED))
        else:
            print(prompt)
            if note:
                print(note)

        hint = "Y/n" if default else "y/N"
        while True:
            raw = input(f"[{hint}] ").strip().lower()
            if not raw:
                return default
            if raw in {"y", "yes"}:
                return True
            if raw in {"n", "no"}:
                return False
            self.print_note("Please answer yes or no.")

    def ask_text(
        self,
        label: str,
        *,
        default: str,
        validator: Callable[[str], bool] | None = None,
        error_message: str = "Invalid value.",
    ) -> str:
        """Prompt for a text value with validation."""
        if self.console is not None:
            self.console.print(Text(label, style=f"bold {ACCENT}"))
        else:
            print(label)

        while True:
            raw = input(f"[default: {default}] ").strip()
            value = raw or default
            if validator is None or validator(value):
                return value
            self.print_note(error_message)

    def ask_existing_path(
        self,
        label: str,
        *,
        default: Path | None,
        allow_blank: bool,
        error_message: str,
        blank_means_none: bool = False,
    ) -> Path | None:
        """Prompt for an existing file or directory path."""
        default_label = self._display_path(default) if default is not None else ""
        if self.console is not None:
            self.console.print(Text(label, style=f"bold {ACCENT}"))
            if default_label:
                self.console.print(Text(f"Default: {default_label}", style=MUTED))
            if allow_blank and blank_means_none:
                self.console.print(Text("Press Enter to skip this optional path.", style=MUTED))
        else:
            print(label)
            if default_label:
                print(f"Default: {default_label}")
            if allow_blank and blank_means_none:
                print("Press Enter to skip this optional path.")

        while True:
            raw = input("> ").strip()
            if not raw:
                if allow_blank and blank_means_none:
                    return None
                if allow_blank and default is None:
                    return None
                if default is not None:
                    candidate = default
                elif allow_blank:
                    return None
                else:
                    self.print_note(error_message)
                    continue
            else:
                expanded = Path(raw).expanduser()
                candidate = expanded if expanded.is_absolute() else (Path.cwd() / expanded)
            resolved = candidate.resolve()
            if resolved.exists():
                return resolved
            if allow_blank and not raw:
                return None
            self.print_note(error_message)

    def _display_path(self, path: Path | None) -> str:
        if path is None:
            return ""
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path.resolve())


def _discover_files(search_root: Path, *, allowed_suffixes: set[str]) -> list[Path]:
    if not search_root.exists():
        return []
    return [
        path
        for path in sorted(search_root.iterdir())
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    ]


def _looks_like_google_contacts_filename(filename: str) -> bool:
    normalized = filename.strip().lower()
    return "google" in normalized or "contacts" in normalized
