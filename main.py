import os
import json
import torch
import getpass
import pickle
import bcrypt
from google.colab import drive
from torch.utils.data import Dataset, DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# 3. Mount Google Drive
drive.mount('/content/drive')

# 4. Paths
DATASET_PATH = '/content/drive/MyDrive/datasets.json'
ADMIN_CREDS_PATH = '/content/drive/MyDrive/banglish_admin_creds.pkl'
MODEL_DIR = '/content/drive/MyDrive/banglish_mbart_model'

SRC_LANG = "en_XX"  # For Banglish/romanized input
TGT_LANG = "en_XX"  # For Banglish/romanized output; for Bangla script use "bn_IN"

# 5. Admin Authentication
def setup_admin_credentials():
    if not os.path.exists(ADMIN_CREDS_PATH):
        print("Setting up admin credentials...")
        username = input("New admin username: ")
        password = getpass.getpass("New admin password: ")
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        with open(ADMIN_CREDS_PATH, 'wb') as f:
            pickle.dump({'username': username, 'hashed': hashed}, f)
        print("Admin credentials saved.")
    return True

def authenticate_admin():
    if not os.path.exists(ADMIN_CREDS_PATH):
        setup_admin_credentials()
    with open(ADMIN_CREDS_PATH, 'rb') as f:
        creds = pickle.load(f)
    username = input("Admin username: ")
    password = getpass.getpass("Admin password: ")
    if username == creds['username'] and bcrypt.checkpw(password.encode('utf-8'), creds['hashed']):
        print("Admin authenticated successfully.")
        return True
    else:
        print("Authentication failed.")
        return False

# 6. Dataset Creation (Sample 10+ entries for test, expand for real)
def create_sample_dataset():
    sample_data = [
        {
            "prompt": "valobasi",
            "result": "bhalobashi",
            "meta-data": "manual"
        },
        {
            "prompt": "asis",
            "result": "achis",
            "meta-data": "manual"
        },
        {
            "prompt": "tako",
            "result": "thako",
            "meta-data": "automatic"
        }
    ]
    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    print(f"Sample dataset created at {DATASET_PATH}")

# 7. Custom Dataset Class
class BanglishDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = []
        for item in data:
            # Now prompt and result are always strings (pairs)
            self.data.append({
                "prompt": item["prompt"],
                "result": item["result"],
                "meta-data": item.get("meta-data") or item.get("collection_type", "unknown")
            })
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["prompt"]
        target_text = item["result"]
        input_encoding = self.tokenizer(
            input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
        )
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Mask pad tokens
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }

# 8. Load Dataset
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        create_sample_dataset()
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 9. Training Function (MBART)
def train_model(epochs=20, src_lang=SRC_LANG, tgt_lang=TGT_LANG):
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer.src_lang = src_lang
    data = load_dataset()
    data = data[:10]  # For sanity check. Remove or increase for real training.
    train_dataset = BanglishDataset(data, tokenizer, max_length=64)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, loss: {total_loss / len(train_loader):.4f}")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Model and tokenizer saved.")

# 10. Inference
def correct_text(text, model, tokenizer, src_lang=SRC_LANG, tgt_lang=TGT_LANG):
    device = model.device
    model.eval()
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        text, return_tensors='pt', max_length=64, padding='max_length', truncation=True
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
        )
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# 11. Dataset update (self-learning)
def update_dataset(prompt_text, result_text, meta_data='manual'):
    data = load_dataset()
    data.append({
        "prompt": prompt_text,
        "result": result_text,
        "meta-data": meta_data
    })
    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Dataset updated.")

# 12. Main Loop
def main():
    print("Welcome to Banglish Corrector!")
    user_type = ""
    while user_type.lower() not in ["y", "n"]:
        user_type = input("Are you an admin? (y/n): ").strip()
    is_admin = False
    if user_type.lower() == "y":
        is_admin = authenticate_admin()
        if not is_admin:
            print("Continuing as normal user...")
    if not os.path.exists(MODEL_DIR):
        print("Training new model...")
        train_model(epochs=20)
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_DIR)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    src_lang = SRC_LANG
    tgt_lang = TGT_LANG
    print("\nType 'quit' to exit.\n")
    while True:
        text = input("Enter Banglish text to correct (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        corrected = correct_text(text, model, tokenizer, src_lang=src_lang, tgt_lang=tgt_lang)
        print(f"Corrected: {corrected}")
        if is_admin:
            feedback = input("Is the correction correct? (y/n): ")
            if feedback.lower() == 'n':
                correct_text_input = input("Enter the correct text: ")
                update_dataset(text, correct_text_input)
                print("Feedback recorded. Model will be updated on next training.")
        else:
            print("If you are an admin and want to update the dataset, please restart and log in as admin.")

if __name__ == "__main__":
    main()