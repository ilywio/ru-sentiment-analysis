import logging 
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logging.info("Start tg-idf imporoving")
logging.info('reading data')


train = pd.read_csv('../data/output/train.csv')
test = pd.read_csv('../data/output/test.csv')

pipeline = Pipeline(
    [
        ('tfidf', TfidfVectorizer()),
        ('lg', LogisticRegression(max_iter=1000))
    ]
)

param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "lg__C": [0.01, 0.1, 1, 10],
    "lg__penalty": ["l2"],
    "lg__solver": ["liblinear", "lbfgs"],
}

logging.info("Cross val (GridSearchCV)")

grid = GridSearchCV(
    pipeline,
    param_grid = param_grid,
    cv=5,
    n_jobs=1,
    scoring = "f1_weighted"
)
grid.fit(train['text'],train['label'])

logging.info(f"the best mean score: {grid.best_score_}")
logging.info(f"the best params: {grid.best_params_}")

logging.info('Start pred in test')

y_pred = grid.predict(test['text'])


logging.info('Metrics')

print('Accuracy: ', accuracy_score(test['label'],y_pred))
print('Precision: ', precision_score(test['label'],y_pred,average='weighted'))
print('Recall: ', recall_score(test['label'],y_pred,average='weighted'))
print('F1 Score: ', f1_score(test['label'],y_pred,average='weighted'))


logging.info('Save params')

with open('../models/tfidf_adv.pkl', 'wb') as file:
    pickle.dump(grid.best_estimator_.named_steps['tfidf'],file = file)
with open('../models/lr_adv.pkl', 'wb') as file:
    pickle.dump(grid.best_estimator_.named_steps['lg'], file=file)
