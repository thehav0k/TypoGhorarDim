# Banglish Text Correction with mBART

This project implements a deep learning model to detect and correct misspelled Banglish text (romanized Bengali) using the mBART model. It includes admin authentication, self-learning via user feedback, and dataset management in Google Colab, with data stored in Google Drive.

## Project Overview

**Objective:**
- Automatically detect and correct misspelled Banglish text (e.g., `valobasi` → `bhalobashi`).

**Model:**
- Fine-tuned `facebook/mbart-large-50-many-to-many-mmt` for sequence-to-sequence correction.

**Dataset:**
- JSON file (`datasets.json`) with paired incorrect (prompt, as a string) and correct (result) Banglish text, plus meta-data ("manual" or "automatic").
  - Example entry:
    ```json
    {
      "prompt": "Ame",
      "result": "Ami",
      "meta-data": "manual"
    }
    ```

**Features:**
- Secure admin authentication with encrypted credentials (set up on first run).
- Self-learning through admin feedback: admins can update the dataset interactively after each correction.
- Training and inference in Google Colab with Google Drive integration.
- If the model does not exist, it will be trained automatically on first run.
- Both admin and normal user modes supported.

**Environment:**
- Google Colab (free tier, GPU recommended).

## Prerequisites

- Google Colab account with access to a GPU runtime.
- Google Drive with sufficient storage (~5 GB for model and dataset).
- Python packages: `torch`, `transformers`, `bcrypt`, `google.colab`.

## Setup Instructions

**Clone the Repository:**
```bash
git clone https://github.com/your-username/banglish-correction.git
cd banglish-correction
```

**Open Google Colab:**
- Create a new notebook in Google Colab.

**Install Dependencies:**
In a Colab cell, run:
```python
!pip install torch transformers bcrypt
```

**Upload Code:**
- Copy the contents of `main.py` from the repository into a Colab cell, or upload and run it with `%run main.py`.

**Mount Google Drive:**
- The code automatically mounts Google Drive to `/content/drive`.
- Ensure you have ~5 GB free for storing `datasets.json`, `banglish_admin_creds.pkl`, and `banglish_mbart_model`.

## Usage

- On first run, set up admin credentials when prompted.
- The code will train a new model if one does not exist, or load the existing model from Google Drive.
- You will be prompted to enter Banglish text for correction.
- If you are an admin, you can provide feedback after each correction to update the dataset interactively.
- To exit, type `quit` when prompted for input.

## Dataset Format

- Each entry in `datasets.json` should have a string for `prompt`, a `result`, and `meta-data`.
- Example:
  ```json
  {
    "prompt": "Ame",
    "result": "Ami",
    "meta-data": "manual"
  }
  ```

## Feedback and Self-Learning

- Admins can update the dataset directly from the CLI after each correction.
- The dataset is saved to Google Drive and will be used for retraining the model.
- To retrain, delete the model directory and rerun the code.

## Manual: Usage and Troubleshooting

**Usage**

- **Run the Code:**
  - Execute `main.py` in Colab after installing dependencies.
  - Follow prompts for authentication, text input, and feedback.

- **Dataset Management:**
  - Edit `datasets.json` in Google Drive to add new entries.
  - Use the sample dataset as a template or generate new data with scripts.

- **Model Retraining:**
  - Delete the model directory to force retraining with updated `datasets.json`.
  - Adjust `epochs` or `batch_size` in `train_model` for performance tuning.

**Troubleshooting**

- **Memory Errors:**
  - Reduce `batch_size` to 1 in `train_model`.
  - Use a smaller dataset for initial testing.

- **Authentication Issues:**
  - If `banglish_admin_creds.pkl` is corrupted, delete it and rerun to set up new credentials.

- **Slow Training:**
  - Use Colab Pro or a smaller dataset for faster prototyping.
  - Limit epochs to 5–10 for testing.

- **Poor Corrections:**
  - Ensure `datasets.json` has diverse, high-quality entries (10,000+ recommended).
  - Include both word-level (e.g., "valobasi" → "bhalobashi") and sentence-level (e.g., "Ami tomake valobasi" → "Ami tomake bhalobashi") pairs.
  - Retrain with feedback to improve accuracy.

> **Note:** Because there are insufficient datasets and Banglish mistakes are dynamic and complex, it is really hard to train the model effectively. Currently, the model is very minimally trained, as only about 1,000 datasets have been used. If more data becomes available, the model will be retrained and fine-tuned for better performance.

> **Warning:** The model currently responds very poorly and mostly cannot correct Banglish text. It is almost of no practical use until a rich dataset is created and used for training.

## Dataset Contribution

We welcome contributions to `datasets.json`! To contribute:

1. Fork the repository.
2. Add new entries to `datasets.json` in the format:
```json
{"prompt": "incorrect_text", "result": "correct_text", "meta-data": "manual"}
```
3. Submit a pull request (PR) with your changes.

Any modification to `datasets.json` will be reviewed and accepted via PR.

## License

MIT License. See LICENSE for details.
