# TextAnatomy — NLP Pipeline Visualizer

Paste any text and watch it move through every stage of NLP: morphology,
syntax, semantics, pragmatics/discourse, and applications — one Streamlit
dashboard, one tab per pipeline stage.

## File structure

```
textanatomy/
├── app.py                     # Streamlit dashboard (entry point)
├── requirements.txt
├── setup_nltk.py              # one-time NLTK data downloader
├── README.md
└── modules/
    ├── __init__.py
    ├── morphology.py          # Module 2 — stemming, lemmatization, n-grams
    ├── syntax_analysis.py     # Module 3 — POS tagging (spaCy + HMM), CFG parse
    ├── semantics.py           # Module 4 — WordNet ambiguity + Lesk WSD
    ├── pragmatics.py          # Module 5 — coreference resolution (+ fallback)
    └── applications.py        # Module 6 — summarization, sentiment, NER, QA, IR search
```

Every file in `modules/` is fully implemented — the functions each one
exposes are called directly by `app.py`, matching exactly what each tab
needs (see the docstring at the top of each file for details).

## Quick start (macOS)

```bash
cd textanatomy
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python setup_nltk.py
python -m spacy download en_core_web_sm
python -m coreferee install en      # optional — see note below if this fails
streamlit run app.py
```

That's the whole setup. Details and troubleshooting below.

## Detailed setup on macOS

### 1. Confirm Python is available
```bash
python3 --version
```
Any Python 3.10–3.14 works. If you don't have Python at all, install it via
[python.org](https://www.python.org/downloads/) or `brew install python`.

### 2. Create and activate a virtual environment
```bash
cd textanatomy
python3 -m venv .venv
source .venv/bin/activate
```
Your terminal prompt should now start with `(.venv)`. **Every command
below assumes this is active** — if you open a new terminal tab, run
`source .venv/bin/activate` again before doing anything else.

### 3. Install the Python packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
This installs Streamlit, NLTK, spaCy, Transformers, PyTorch, scikit-learn,
pandas, and coreferee. This step downloads several hundred MB (mainly
PyTorch) — it can take a few minutes on a fresh install.

> **If `pip install` fails on a package** (most likely `coreferee`, occasionally
> `spacy` or `torch` on a brand-new Python version): see Troubleshooting below.

### 4. Download NLTK's data files (one-time)
```bash
python setup_nltk.py
```
This grabs `punkt`/`punkt_tab` (tokenization), `wordnet`/`omw-1.4`
(semantics), and `brown` (the HMM tagger's training data). Safe to
re-run any time — already-downloaded resources are skipped.

### 5. Download spaCy's English model
```bash
python -m spacy download en_core_web_sm
```
This is separate from `pip install spacy` — spaCy ships the *library*,
but language models are downloaded separately.

### 6. Download the coreferee coreference model (optional but recommended)
```bash
python -m coreferee install en
```
If this step fails, don't worry — skip it. The Pragmatics tab is built
to automatically fall back to a simpler heuristic if coreferee isn't
available (see Notes below).

## Running the app

```bash
source .venv/bin/activate      # if not already active
streamlit run app.py
```

**Important:** always use `streamlit run app.py` — never `python app.py`
or `python3 app.py`. Streamlit apps are not regular scripts; running one
as a plain script will either error immediately or do nothing useful,
because it needs Streamlit's own server process to render the dashboard.

This opens your browser automatically at `http://localhost:8501`. Paste
in any article, story, or review (or use the pre-filled sample text) and
click **Analyze**.

## What each tab does

| Tab | What it shows |
|---|---|
| **Overview** | Word/character counts and a plain-English summary of the pipeline |
| **Morphology** | Porter stemming vs. WordNet lemmatization, side by side, plus the most frequent bigrams |
| **Syntax** | spaCy POS tags, a from-scratch HMM POS tagger trained live on the Brown corpus, and a hand-written CFG grammar that produces two valid parse trees for one classically ambiguous sentence |
| **Semantics** | Words in your text with genuine WordNet ambiguity, and the Lesk algorithm's guess at which sense is meant in context |
| **Pragmatics** | Coreference chains (e.g. "Ravi" → "he" → "his") via `coreferee`, or a transparent nearest-preceding-noun heuristic if `coreferee` isn't available |
| **Applications** | Summarization, sentiment analysis, named entities, a Q&A box you can type into, and a TF-IDF search box over the text's own sentences |

## Notes

- **First run will be slower** — it downloads the HuggingFace models
  (`distilbart-cnn-12-6`, `distilbert-base-uncased-finetuned-sst-2-english`,
  `distilbert-base-cased-distilled-squad`) and caches them locally
  (`~/.cache/huggingface`). After that, startup is much faster. Same goes
  for the HMM tagger — it's retrained the first time you open the Syntax
  tab in a session, then cached in memory for the rest of that session.
- Everything runs on CPU — no GPU needed. On an M-series Mac this is fast;
  on Intel Macs, summarization/QA may take a few seconds longer per query.
- If `coreferee` fails to install, load, or find any chains, the Pragmatics
  tab automatically falls back to a simple recency-based pronoun-resolution
  heuristic, so the app never breaks — and this actually gives you a nice
  model-vs-heuristic comparison point for a report.
- Every analysis block in `app.py` is wrapped in `try/except`, so if one
  library has a hiccup on your machine, the rest of the dashboard still works.

## Troubleshooting

| Error you see | What it means / how to fix it |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` (or streamlit/spacy/etc.) | Your virtual environment isn't active, or packages were installed into a *different* Python. Run `source .venv/bin/activate` first, then `pip install -r requirements.txt` again — confirm with `which python` that it points inside `.venv`. |
| `ModuleNotFoundError: No module named 'modules'` | You're running the app from the wrong folder. `cd` into the `textanatomy` folder itself (the one containing `app.py`) before running `streamlit run app.py`. |
| `LookupError: Resource punkt_tab not found` (or wordnet/brown/etc.) | You skipped or need to re-run `python setup_nltk.py`. Run it again — it's safe to repeat. |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | You skipped `python -m spacy download en_core_web_sm`. Run it (with the venv active). |
| `pip install coreferee` fails, or `python -m coreferee install en` fails | Expected on some Python/spaCy version combinations — `coreferee` is a small, infrequently-updated package. Comment out the `coreferee` line in `requirements.txt` and skip step 6 entirely; the Pragmatics tab's built-in fallback handles this automatically. |
| `pip install` fails while building `torch` or `spacy` on a brand-new Python version | Very new Python releases can briefly lack pre-built wheels for some packages. Install Python 3.11 via `brew install python@3.11`, recreate the venv with `python3.11 -m venv .venv`, and reinstall — this exact combination is the most widely-tested one for this stack. |
| Warning: `Your max_length is set to 142, but your input_length is only ...` | Handled already — `summarize_text()` scales `max_length`/`min_length` to the input size, so this shouldn't appear. If it does on unusually short text, it's a harmless warning, not an error. |
| Browser doesn't open automatically | Open `http://localhost:8501` manually. |
| `Port 8501 is already in use` | Another Streamlit app is already running. Either stop it (`Ctrl+C` in its terminal), or run `streamlit run app.py --server.port 8502`. |
| Everything installed but the app feels slow on first analysis | Normal — the first click triggers spaCy model load + HMM training + HuggingFace model downloads all at once. Subsequent clicks in the same session are much faster. |
