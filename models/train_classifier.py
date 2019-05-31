                ### imports ###

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'wordnet'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.corpus import stopwords

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

import warnings
warnings.simplefilter('ignore')

import subprocess
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
install('termcolor')
from termcolor import colored, cprint


                ### Functions ###

def load_data(database_filepath):
    """
    Load and merge datasets

    input:
        database_filename: Filename for a clean database

    outputs:
        X: messages dataset
        y: all other features dataset
        category_names: List with all category names.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    # listing the columns
    category_names = list(np.array(y.columns))

    return X, y, category_names

def tokenize(text):
    """
    Normalize, tokenize and lemmatize text string

    input:
        text: messages

    output:
        clean_tokens: clean data
    """
    # Normalize text
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:

        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Build a pipeline

    input:
        None

    output:
        cv: GridSearch model with pipeline and classifier
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # specify parameters for grid search
    parameters = {'vect__min_df': [1, 5],
                'tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [10, 25],
                'clf__estimator__min_samples_split': [2, 4]}

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    prints classifaction_report with precision, recall, and F-score

    inputs:
        model: model object
        X_test
        y_test
        category_names: List of all category names.

    output:
        None
    """
    # predict on test data
    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(y_test[col],
                                                                    y_pred[:, i],
                                                                    average='weighted')

        print('\nReport for the column ({}):\n'.format(colored(col, 'blue', attrs=['bold', 'underline'])))

        if precision >= 0.75:
            print('Precision: {}'.format(colored(round(precision, 2), 'green')))
        else:
            print('Precision: {}'.format(colored(round(precision, 2), 'red')))

        if recall >= 0.75:
            print('Recall: {}'.format(colored(round(recall, 2), 'green')))
        else:
            print('Recall: {}'.format(colored(round(recall, 2), 'red')))

        if fscore >= 0.75:
            print('F-score: {}'.format(colored(round(fscore, 2), 'green')))
        else:
            print('F-score: {}'.format(colored(round(fscore, 2), 'red')))

def save_model(model, model_filepath):
     """
     Save model as a pickle file

    inputs:
        model: model object
        model_filepath: Filepath to save the model

    output:
        None
    """
    # Open the file to save as pkl file
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
