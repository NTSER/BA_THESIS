import re
import csv

from datetime import datetime
from itertools import product

import pandas as pd

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def preprocess_document(text):
    text = re.sub(r'[^ა-ჰ ]', ' ', text)
    text = re.sub(r' +', ' ', text)
    list_of_words = text.split()
    return list_of_words

def tag_documents(list_of_documents):
    for i, document in enumerate(list_of_documents):
        list_of_words = preprocess_document(document)
        yield TaggedDocument(list_of_words, [i])

def evaluate(model, train_idx, test_idx, y_train, y_test):
    logit = LogisticRegression(max_iter=100000)
    X_train = model.dv.vectors[train_idx]
    X_test = model.dv.vectors[test_idx]
    logit.fit(X_train, y_train)
    pred = logit.predict(X_test)
    return f1_score(y_test, pred, average='macro')
        
class EarlyStopException(Exception):
    pass

class EarlyStoppingCallback(CallbackAny2Vec):
    def __init__(self, patience=5):
        self.patience = patience
        self.current_f1_score = 0
        self.best_f1_score = 0
        self.counter = 0
        
    def on_epoch_end(self, model):
        self.current_f1_score = evaluate(model, train_df.index, test_df.index, y_train, y_test)

        if self.current_f1_score > self.best_f1_score:
            self.best_f1_score = self.current_f1_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered. Training stopped.")
                raise EarlyStopException()

        
df = pd.read_csv('../political_news.csv')
train_df, test_df = train_test_split(df, test_size=0.03, random_state=7, stratify=df['source'])
y_train = train_df['source']
y_test = test_df['source']

corpus = list(tag_documents(df['content']))

param_space = {
    'vector_size':[1024, 512],
    'window':[5],
    'alpha':[0.025],
    'min_alpha':[0.001],
    'max_vocab_size':[10**4]
}

#Initialize training_logs.csv with only column-names
with open('training_logs.csv', "w", encoding="utf-8",  newline="") as csv_file:
    columns = ['vector_size', 'window', 'alpha', 'min_alpha','max_vocab_size', 'best_f1_score']
    writer = csv.writer(csv_file)
    writer.writerow(columns)


combinations = list(product(*param_space.values()))
for comb in combinations:
    current_params = {k: v for k, v in zip(param_space.keys(), comb)}
    model = Doc2Vec(**current_params)
    model.build_vocab(corpus)

    earlyStoppingCallback = EarlyStoppingCallback()

    try:
        model.train(corpus, total_examples=model.corpus_count, epochs=1000,
                    callbacks=[earlyStoppingCallback])

    except EarlyStopException:
        pass

    finally:
        with open("training_logs.csv", "a", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            stats = list(current_params.values()) + [earlyStoppingCallback.best_f1_score]
            writer.writerow(stats)
        print(f'stats: {stats}, current_time: {datetime.now().strftime("%H:%M:%S")}')


# Train best model
logs_df = pd.read_csv('training_logs.csv')
best_hyperparams = (logs_df
                    .sort_values(by='best_f1_score', ascending=False)
                    .drop('best_f1_score',axis=1)
                    .to_dict(orient='records')[0])

model = Doc2Vec(**best_hyperparams)
model.build_vocab(corpus)

earlyStoppingCallback = EarlyStoppingCallback()

try:
    model.train(corpus, total_examples=model.corpus_count, epochs=1000,
                callbacks=[earlyStoppingCallback])

except EarlyStopException:
    pass

finally:
    model.save('bestmodel.model')
    print(f'Best model saved')
