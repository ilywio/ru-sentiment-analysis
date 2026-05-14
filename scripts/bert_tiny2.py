from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import logging
import pandas as pd 
from datasets import Dataset, DatasetDict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


TEXT2ID = {"негативная": 0, "позитивная": 1, "мусор": 2, "нейтральная": 3}


logging.basicConfig(level = logging.INFO)
logging.info("Start train (Bert - 'cointegrated/rubert-tiny2')")
logging.info('Read data')
train = pd.read_csv('../data/output/train.csv')
test = pd.read_csv('../data/output/test.csv')

train['label'] = train['label'].map(TEXT2ID)
test['label'] = test['label'].map(TEXT2ID)


logging.info('Prepare model')
model_name = "cointegrated/rubert-tiny2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels = 4
)

def tokenize(batch):
    return tokenizer(batch['text'], truncation = True)

dataset = DatasetDict({
    "train": Dataset.from_dict({"text": train['text'], "label": train['label']}),
    "test": Dataset.from_dict({"text": test['text'], "label": test['label']}),
})


dataset = dataset.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

logging.info('Training')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_args = TrainingArguments(
    output_dir = '../models/bert_results',
    learning_rate = 2e-5,
    per_device_train_batch_size = 16,
    num_train_epochs = 3,
    eval_strategy = 'epoch',
    save_strategy="epoch"
)


trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics = compute_metrics,
    data_collator = data_collator
)

trainer.train()

trainer.evaluate()




