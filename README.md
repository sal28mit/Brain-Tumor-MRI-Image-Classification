# Brain Tumor MRI Image Classification

Deep-learning pipeline that classifies brain MRI images into tumour types (custom CNN from scratch + transfer learning with
MobileNetV2, EfficientNetB0 and ResNet50) and serves the best model through a Streamlit web app.

> **Disclaimer:** research / education prototype - **not a medical device**. Every prediction must be confirmed by a qualified radiologist.

## Project structure

```
Brain_Tumor_MRI_Project/
|-- Brain_Tumor_MRI_Classification.ipynb   # MAIN notebook (follows the ML submission template): EDA, hypothesis tests,
|                                          #   preprocessing, 4 models, tuning, comparison, explainability, saving
|-- Streamlit_Deployment.ipynb             # writes app.py, tests the model, launches the web app from Jupyter
|-- app.py                                 # Streamlit application (same file the deployment notebook writes)
|-- requirements.txt
|-- Brain_Tumor_MRI_Dataset/               # <-- YOU put the dataset here (not included)
|-- models/            (created)           # custom_cnn.h5, mobilenetv2.h5, efficientnetb0.h5, resnet50.h5, best_model.h5,
|                                          #   model_config.json, class_names.json
|-- outputs/           (created)           # every chart (PNG) + model_comparison.csv
|-- data/              (created)           # image_metadata.csv, brain_tumor_mri.db (SQLite), prediction_history.db (app)
`-- checkpoints/       (created)           # best-weight checkpoints during training
```

## Quick start

```bash
# 1. (recommended) new virtual environment, Python 3.10 - 3.12
python -m venv .venv
.venv\Scripts\activate            # Windows      |  source .venv/bin/activate   (macOS / Linux)

# 2. install
pip install -r requirements.txt

# 3. dataset: download the Drive folder / zip and extract it into  ./Brain_Tumor_MRI_Dataset
#    (a .zip placed next to the notebook is also extracted automatically)

# 4. train + evaluate everything
jupyter notebook Brain_Tumor_MRI_Classification.ipynb      # Kernel -> Restart & Run All

# 5. deploy
jupyter notebook Streamlit_Deployment.ipynb                # Run All   (or:  streamlit run app.py)
```

## Dataset

Source: *Brain Tumor MRI Multi-Class Dataset* (Google Drive link in the project brief). The notebook **auto-detects** the layout:
`train/valid/test/<class>/`, `Training/Testing/<class>/` or just `<class>/` (a stratified 70/15/15 split is created when no official
split exists). Class-folder names are harmonised if the splits spell them differently. Corrupted files, exact duplicates and
train/test leakage (MD5 fingerprints) are removed.

> Google Drive cannot download large image folders reliably from code, so download it manually (as a zip) and extract it.

There is no ready-made database in this project; the notebook **creates** one: `data/image_metadata.csv` and the SQLite database
`data/brain_tumor_mri.db` (tables `image_metadata` and `model_results`). The app keeps its own `prediction_history.db`.

## Runtime and switches (Configuration cell of the main notebook)

| Setting | Default | Meaning |
|---|---|---|
| `QUICK_TEST` | `False` | `True` = tiny subset + 2 epochs: runs the whole notebook in a few minutes to verify your setup |
| `MODELS_TO_TRAIN` | all three | e.g. `["MobileNetV2"]` to train only one transfer-learning model |
| `RUN_CNN_TUNING` | `True` | random hyper-parameter search for the custom CNN |
| `EPOCHS_CNN / HEAD / FT` | 25 / 8 / 10 | upper limits - EarlyStopping usually stops earlier |
| `BATCH_SIZE` | 32 | lower it (16) if you run out of memory |

A GPU (or Google Colab) is strongly recommended for the full run; on a CPU use `QUICK_TEST` first and expect a long full run.

## What the notebook does

1. **Know your data / variables / wrangling** - file index, metadata (size, brightness, contrast, MD5), duplicates, corrupted files.
2. **15 charts** (UBM rule), each with *why / insight / business impact*.
3. **3 hypothesis tests** - Kruskal-Wallis, Chi-square goodness-of-fit, Mann-Whitney U.
4. **Pre-processing** - RGB, resize 224x224, scale 0-1, augmentation (flip, rotation, zoom, shift, contrast, brightness), PCA view, class weights.
5. **Models** - Custom CNN (+ random search), MobileNetV2, EfficientNetB0, ResNet50 (frozen -> fine-tuned). EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
6. **Evaluation** - accuracy, precision, recall, F1 (macro / weighted), ROC-AUC, worst-class recall, confusion matrices, training curves.
7. **Comparison + selection** on the *validation* set (test set stays unbiased); size / speed included.
8. **Explainability** (occlusion sensitivity), **save** `.h5` models + JSON config, **reload and sanity-check**.

## Streamlit app features

Multi-image upload - model selector (best model pre-selected) - predicted class + confidence + probability chart for every class -
low-confidence warning (adjustable threshold) - batch summary with CSV download - model-comparison tab - SQLite prediction history.

## Troubleshooting

* **`Dataset not found`** - place the extracted dataset in `Brain_Tumor_MRI_Dataset/` next to the notebook (or change `DATA_DIR`).
* **ImageNet weights cannot be downloaded** (no internet) - the notebook prints a warning and trains that model from scratch instead of failing.
* **Out-of-memory** - reduce `BATCH_SIZE`; close other programs; use a GPU runtime.
* **`.h5` save/load warnings in newer Keras** - harmless; if `.h5` saving ever fails the notebook falls back to `.keras`, and the app loads either.
* **Port 8501 busy** - change `PORT` in the deployment notebook or run `streamlit run app.py --server.port 8502`.

## Publishing to GitHub

```bash
git init && git add . && git commit -m "Brain tumor MRI classification"
git branch -M main
git remote add origin https://github.com/<your-username>/brain-tumor-mri-classification.git
git push -u origin main
```
`.gitignore` already excludes the dataset and checkpoints. ResNet50 / EfficientNet `.h5` files can exceed GitHub's 100 MB limit -
use [Git LFS](https://git-lfs.com) (`git lfs track "*.h5"`) or commit only `best_model.h5` plus the smaller models.
