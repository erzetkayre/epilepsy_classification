# Epilepsy Classification from EEG Signals

Final project code for classifying epileptic seizures from EEG
signals using deep learning. Signals are drawn from the University of Bonn
(UKB) EEG dataset and classified with CNN, LSTM, GRU, and hybrid CNN-LSTM /
CNN-GRU architectures, for both binary (healthy vs. epileptic) and ternary
(healthy / interictal / ictal) classification.

## Project structure

```
code/
├── data/                       # raw and processed EEG data (gitignored)
├── notebooks/                  # experiment notebooks, in run order
│   ├── 01_preprocessing/
│   ├── 02_cnn_model_development/
│   ├── 03_lstm_model_development/
│   ├── 04_gru_model_development/
│   ├── 05_hybrid_cnn_lstm_model_development/
│   └── 06_hybrid_cnn_gru_model_development/
├── py/                          # standalone experiment scripts
├── best model/                  # trained model checkpoints (gitignored)
├── requirements.txt
└── .gitignore
```

Each model-development notebook folder is split into `binary_classification`
and `ternary_classification` subfolders. Only the notebooks (`.ipynb`) are
tracked in git; generated artifacts inside them (plots, training logs,
CV results) are not.

## Setup

```bash
python -m venv env
env\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Data

EEG data is not tracked in git. Place the raw UKB text files under
`data/ukb_raw/` (sets `Z`, `O`, `N`, `F`, `S`, per Andrzejak et al. 2001)
before running the preprocessing notebook in `notebooks/01_preprocessing/`.

## Usage

Run the notebooks in numeric order:

1. `01_preprocessing` — load, filter, and label the raw EEG signals.
2. `02`–`06` — train and evaluate each model architecture on the
   preprocessed data, for both binary and ternary classification.
