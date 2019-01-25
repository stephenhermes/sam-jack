"""Train a NLP model using WOrd2Vec."""

import datetime
import pathlib
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('seaborn')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, precision_score, 
    recall_score, roc_auc_score, roc_curve, log_loss, precision_recall_curve
)

from scipy.stats import randint

import nltk
import gensim


class WordVectorizer(TransformerMixin, BaseEstimator):
    """Preprocess text into word vectors.
    
    Note: The option to train word vectors from scratch is not 
    yet implemented.
        
    Parameters
    ----------
    model : Gensim model, optional (default=None)
        The model to use for encoding word vectors. If None then 
        will train word embeddings from scratch during training.
        
    stopwords : list of strings, optional (default=None)
        A list of stop words to remove from input.
        
    tokenize : bool, optional (default=True)
        Whether or not input should be tokenized. Shouls be set
        to False if being used with a different tokenization
        algorithm.
        
    """
    def __init__(self, model=None, stopwords=None, tokenize=True):
        
        self.model = model
        self.stopwords = stopwords
        self.tokenize = tokenize
        if self.model is None:
            # Instantiate gensim.model.Word2Vect and params
            raise NotImplementedError(
                   "Custom word vector training not implemented."
            )
        
    def fit(self, X, y=None):
        if self.model is not None:
            self.vectorizer_ = self.model
        else:
            # Train gensim
            pass
        return self
    
    def transform(self, X, y=None):
        stopwords = []
        if self.stopwords is not None:
            stopwords = self.stopwords
        
        def _tokenize(string):
            """Tokenize single input string."""
        
            string = string.replace('\n', ' ').lower()
            tokens = [
                w for sent in nltk.tokenize.sent_tokenize(string)
                for w in nltk.tokenize.word_tokenize(sent)
                if w not in stopwords
            ]
            return tokens
        
        def _vectorize(tokens):
            """Replace tokens with word vectors from model.

            Words not found in the model keywords populated by
            array of nans.
            
            """
            model = self.vectorizer_
            default_val = np.full(model.vector_size, np.nan)
            vectors = [
                model.get_vector(t) if t in model.vocab else default_val
                for t in tokens
            ]
            return np.array(vectors)
        
        X_tokenized = X
        if self.tokenize: 
            X_tokenized = [_tokenize(string) for string in X]
        X_vectorized = [_vectorize(t) for t in X_tokenized]
        X_vectorized = np.array([np.nanmean(v, axis=0) for v in X_vectorized])
        
        return X_vectorized


def make_classifier(**params):
    """Instantiate classifier.
    
    Parameters
    ----------
    model : Gensim model, optional (default=None)
        The model to use for encoding word vectors. If None then 
        will train word embeddings from scratch during training.
        
    stopwords : list of strings, optional (default=None)
        A list of stop words to remove from input.


    Returns
    -------
    classifier : sklearn RandomizedSearchCV object
        A tuneable multi-label classification model.
    
    """
    model = params.pop('model', None)
    stopwords = params.pop('stopwords', None)
    n_estimators = params.pop('n_estimators', 100)
    n_jobs = params.pop('n_jobs', 1)
    cv = params.pop('cv', 3)
    n_iter = params.pop('n_iter', 10)
    scoring = params.pop('scoring', 'roc_auc')
    verbose = params.pop('verbose', 1)

    estimator = make_pipeline(
        WordVectorizer(model=model, stopwords=stopwords),
        RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=42, 
            n_jobs=n_jobs
        )
    )

    # Parameters to tune
    params = {
        'randomforestclassifier__min_samples_split': randint(3, 20),
        'randomforestclassifier__min_samples_leaf': randint(1, 20),
        'randomforestclassifier__max_depth': randint(3, 20),
    }
    # Tuneable classifier
    classifier = RandomizedSearchCV(
        estimator, params,
        cv=cv, n_iter=n_iter, random_state=42, n_jobs=n_jobs,
        scoring=scoring, verbose=verbose,
        error_score=-1
    )

    return classifier


def multilabel_accuracy(y_true, y_pred, weights=None):
    """Compute the multi-label accuracy."""
    matches = (np.around(y_true) == np.around(y_pred)).astype(np.float32)
    class_means = matches.mean(axis=0)
    if weights is None:
        # Classes given equal weight by default
        weights = np.ones_like(class_means) / len(class_means)
    assert len(weights) == len(class_means)
    weights = weights / weights.sum()  # Normalize
    return (weights * class_means).sum()



if __name__ == '__main__':
    project_path = pathlib.Path(__file__).resolve().parents[2]
    
    # Load film data
    labels_raw = pd.read_csv(
        project_path / 'data' / 'processed' / 'genres_labels.csv', 
        index_col=[0, 1]
    )
    plots_raw = pd.read_csv(
        project_path / 'data' / 'processed' / 'film_plots.csv', 
        index_col=[0, 1]
    )

    label_names = labels_raw.columns.values
    labels = labels_raw.values
    plots = plots_raw.values.flatten()

    # Load nlp data
    stopwords = nltk.corpus.stopwords.words('english')
    model = gensim.models.KeyedVectors.load_word2vec_format(
        project_path / 'data' / 'external' / 
            'GoogleNews-vectors-negative300.bin', 
        binary=True
    )
    n_features = model.vector_size

    # Build model
    classifier = make_classifier(
        model=model, 
        stopwords=stopwords,
        n_estimators=1,
        n_jobs=4, 
        cv=2, 
        n_iter=1
        )

    # Tune model
    corpus_train, corpus_test, labels_train, labels_test = \
        train_test_split(plots, labels, test_size=0.2, random_state=42)

    classifier.fit(corpus_train, labels_train)

    # Print model loss
    y_pred_train_raw = classifier.predict_proba(corpus_train)
    y_pred_train = np.array(y_pred_train_raw)[:, :, 1].transpose()
    y_pred_test_raw = classifier.predict_proba(corpus_test)
    y_pred_test = np.array(y_pred_test_raw)[:, :, 1].transpose()
    print(
        'train: {:.5f}'.format(log_loss(labels_train, y_pred_train)),
        '\ntest: {:.5f}'.format(log_loss(labels_test, y_pred_test))
    )

    # Print model accuracy
    weights = labels_train.sum(axis=0)
    print(
        'macro weighted average accuracy train: {:.5f}'.format(
            multilabel_accuracy(labels_train, y_pred_train, weights)
        ),
        '\nmacro weighted average accuracy test: {:.5f}'.format(
            multilabel_accuracy(labels_test, y_pred_test, weights)
        )
    )
    
    # Fit on full data and serialize
    estimator = classifier.best_estimator_
    estimator.fit(plots, labels)
    coda = datetime.datetime().now().strftime('%Y%m%d_%h%m%s')
    with open(project_path / 'models' / f'word2vect_{coda}.pkl', 'wb') as f:
        pickle.dump(estimator, f)
