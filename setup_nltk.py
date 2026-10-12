"""
Run this once after installing requirements.txt:
    python setup_nltk.py

Downloads every NLTK data package the project needs. Wrapped in
try/except per-item because resource names differ slightly across
NLTK versions (e.g. punkt vs punkt_tab) - safe to ignore failures
for names that don't apply to your installed version.
"""

import nltk

RESOURCES = [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "omw-1.4",
    "brown",
    "universal_tagset",
]

if __name__ == "__main__":
    for resource in RESOURCES:
        try:
            nltk.download(resource)
        except Exception as exc:
            print(f"Skipped '{resource}': {exc}")

    print("\nNLTK setup complete.")
    print("Next steps (run in terminal, not this script):")
    print("  python -m spacy download en_core_web_sm")
    print("  python -m coreferee install en")
