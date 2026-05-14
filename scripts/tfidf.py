from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
import pandas as pd 

from pathlib import Path
import logging
from scipy.sparse import vstack
import pickle

logging.basicConfig(level=logging.INFO)
logging.info('Reading train/test datasets')

output_path = Path('../data/output')

train_df = pd.read_csv(output_path/'train.csv')
test_df = pd.read_csv(output_path/'test.csv')

logging.info('Trainig tf-idf')
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(train_df['text'])

X_test_tfidf = vectorizer.transform(test_df['text'])

logging.info('Train log regression')
model = LogisticRegression(max_iter=200)

model.fit(X_train_tfidf,train_df['label'])

y_pred = model.predict(X_test_tfidf)

logging.info('Metrics')

print('Accuracy: ', accuracy_score(test_df['label'],y_pred))
print('Precision: ', precision_score(test_df['label'],y_pred,average='weighted'))
print('Recall: ', recall_score(test_df['label'],y_pred,average='weighted'))
print('F1 Score: ', f1_score(test_df['label'],y_pred,average='weighted'))


logging.info('Cross-Validation')
X_all = vstack([X_train_tfidf, X_test_tfidf])
df = pd.concat([train_df,test_df])
scores = cross_val_score(model,X_all,df['label'],cv = 5,scoring = 'f1_weighted' ) 

print (scores)
print("Mean (f1):", scores.mean())

logging.info('Save model')
with open('../models/tfidf_start.pkl', 'wb') as file:
    pickle.dump(vectorizer,file = file)
with open('../models/lr_start.pkl', 'wb') as file:
    pickle.dump(model, file=file)

