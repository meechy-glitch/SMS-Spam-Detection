import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = "model/distilbert-spam"
MAX_LEN = 128


def load_data():
    df = pd.read_csv("data/spam.csv", encoding="latin-1", usecols=["v1", "v2"])
    df.columns = ["label", "message"]
    df["label_id"] = df["label"].map({"ham": 0, "spam": 1})
    return df


class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=MAX_LEN
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def main():
    df = load_data()
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_id"]
    )

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Spam ratio — Train: {train_df['label_id'].mean():.2%} | Test: {test_df['label_id'].mean():.2%}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = SMSDataset(train_df["message"].tolist(), train_df["label_id"].tolist(), tokenizer)
    test_dataset = SMSDataset(test_df["message"].tolist(), test_df["label_id"].tolist(), tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    os.makedirs(f"{SAVE_PATH}/final", exist_ok=True)
    model.save_pretrained(f"{SAVE_PATH}/final")
    tokenizer.save_pretrained(f"{SAVE_PATH}/final")
    print(f"\nModel saved to {SAVE_PATH}/final")

    results = trainer.evaluate()
    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {results['eval_accuracy']:.4f}")
    print(f"Precision: {results['eval_precision']:.4f}")
    print(f"Recall:    {results['eval_recall']:.4f}")
    print(f"F1 Score:  {results['eval_f1']:.4f}")

    preds_output = trainer.predict(test_dataset)
    preds = preds_output.predictions.argmax(-1)
    labels = test_df["label_id"].tolist()
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"           Ham    Spam")
    print(f"Ham      {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"Spam     {cm[1][0]:5d}   {cm[1][1]:5d}")


if __name__ == "__main__":
    main()
