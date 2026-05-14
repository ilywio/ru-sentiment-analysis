from pandas._config import config
from tqdm import tqdm
from pathlib import Path
import logging 
import pandas as pd

logging.basicConfig(level=logging.INFO)
logging.info("Start reading file")

path = Path("../data/raw/data_raw.xlsx")
sheets = pd.read_excel(path, sheet_name=None)

print(sheets.keys())

del sheets["Лист1"]
dfs = []

print(sheets.keys())


for name, df in tqdm(sheets.items(), desc = "Reading lists"):

    df = df.copy()

    df = df[["Комментарии", "Эмоциональная окраска"]]

    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

final_df.to_csv('../data/interim/data_read.csv', index=False)

logging.info('Done')
