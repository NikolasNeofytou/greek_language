# Greek Language Web Application

This project provides a minimal web application for analyzing Greek words. The first feature extracts the lemma and basic morphological information for a given word using spaCy's Greek model.

The application now also provides an English translation for each token using a transformer-based model. A helper script is included for fine-tuning a small Greek language model.

## Setup

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python app.py
```

Navigate to `http://localhost:5000` and enter a Greek word to see its lemma and morphological analysis.

## Training a Greek Language Model

A simple training script is provided to fine-tune GPT‑2 on a small subset of Greek text from the OSCAR corpus:

```bash
python train_llm.py --output_dir greek-llm
```

The script downloads the dataset and saves the fine-tuned model to the specified directory. You can replace the translation model in `app.py` with your fine-tuned model for customized behaviour.
