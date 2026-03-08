# Contact Name Recovery Agent

A local-first, privacy-aware Python agent for recovering names from phone numbers using user-provided contact files, optional public web snippets, and a local Ollama model for reasoning.

## What it does

- Loads phone numbers from `.xlsx`, `.csv`, `.txt`, `.json`, or `.vcf`
- Normalizes numbers to E.164 with `phonenumbers`
- Checks direct matches from a local Google Contacts CSV or vCard
- Uses a local Ollama model to plan next actions and reflect on noisy evidence
- Optionally performs quoted-number public web search only when `--allow-web yes`
- Asks the user for clarification when confidence stays low
- Stores resolved contacts in local JSON memory for future runs
- Exports results to `output/recovered.csv`

## Privacy defaults

- Core logic stays local
- Web search is off by default
- `memory/session_history.json` stores masked phone numbers only
- `memory/resolved_contacts.json` stores raw E.164 numbers locally because the agent needs them for future matching

## Folder layout

```text
contact_name_agent/
├── agent.py
├── tools.py
├── memory.py
├── utils.py
├── prompts.py
├── __init__.py
├── inputs/
│   └── numbers.csv
├── memory/
│   ├── resolved_contacts.json
│   └── session_history.json
├── output/
├── requirements.txt
└── README.md
```

## Input formats

### `inputs/numbers.csv` or `inputs/numbers.xlsx`

Preferred header:

```csv
phone
+919876543210
```

Accepted alternative headers include `number`, `mobile`, `telephone`, and `contact`.

Excel workbooks are also supported. The agent will scan all sheets and look for phone-like columns automatically.

### Optional Google Contacts file

Pass a file with:

```bash
python agent.py run --input inputs/numbers.csv --google-contacts inputs/google_contacts.csv
```

The agent supports:

- `.xlsx` workbooks
- Google Contacts CSV exports
- `.vcf` / vCard files

## Usage

Launch the guided onboarding flow:

```bash
python agent.py onboard
```

The onboarding wizard walks through:

- privacy and local-storage defaults
- input file selection from `inputs/` or a custom path
- optional Google Contacts file selection
- default region setup
- safe/local vs opt-in web search
- optional Truecaller usage when locally configured
- final review before the run starts

Run in fully local mode:

```bash
python agent.py run --input inputs/numbers.csv --model llama3.2:latest --allow-web no
```

On the first interactive run, the agent asks once for your default country or dialing code, saves it in `memory/preferences.json`, and reuses it on future runs.

Run with opt-in public search:

```bash
python agent.py run --input inputs/numbers.csv --model llama3.2:latest --allow-web yes
```

## Notes

- If Ollama is unavailable, the agent falls back to deterministic heuristics instead of crashing.
- If web search packages are unavailable, the agent logs the issue and continues locally.
- `resolved_contacts.json` is treated as the persistent memory source of truth for future runs.
- The saved default parsing region lives in `memory/preferences.json`

## Optional Truecaller CLI

The project can also use the vendored `truecaller-cli` wrapper as an optional source for Indian mobile numbers.

Setup:

```bash
python agent.py setup-truecaller --number YOUR_10_DIGIT_NUMBER
```

If OTP registration fails but you already have a valid Truecaller `installationId`, save it manually:

```bash
python agent.py setup-truecaller-token --installation-id YOUR_INSTALLATION_ID
```

Run with Truecaller enabled:

```bash
python agent.py run --input inputs/unsaved_134_numbers.xlsx --allow-web no
```

Notes:

- This requires the vendored Node dependencies in `_vendor/truecaller-cli`
- The Truecaller CLI needs OTP registration first
- Truecaller lookup is enabled by default once registration exists

Quick Start
```bash
pip install -r requirements.txt
ollama pull llama3.2:latest   # or your preferred model
python agent.py onboard       # guided setup with review panels
python agent.py run --input inputs/numbers.csv --allow-web no   # start safe/local-only
```
