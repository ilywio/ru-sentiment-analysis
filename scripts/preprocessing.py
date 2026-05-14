import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logging.info("Start reading file")

data_path = Path("../data/interim/data_read.csv")
df = pd.read_csv(data_path)

logging.info("Rename columns")
dict_columns = {"Комментарии": "text", "Эмоциональная окраска": "label"}
df.rename(columns=dict_columns, inplace=True)
df.loc[df["label"] == "позитивная", "label"] = "Позитивная"

logging.info("Drop missing values")
df.dropna()

logging.info("Drop outliers")
q = df["text"].str.len().quantile(0.99)
df = df[df["text"].str.len() < q]

logging.info("Label normalization (lowercase, delete spaces)")

df["label"] = df["label"].str.strip().str.lower()

logging.info("Split dataset")
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.9, stratify=y, shuffle=True, random_state=42
)

logging.info("Save train/test datasets to output dir")
train_df = pd.DataFrame({"text": X_train, "label": y_train})
test_df = pd.DataFrame({"text": X_test, "label": y_test})

output_path = Path("../data/output")

train_df.to_csv(output_path / "train.csv")
test_df.to_csv(output_path / "test.csv")
