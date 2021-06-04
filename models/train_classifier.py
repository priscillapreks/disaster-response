# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re 
import pickle

import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''Loads data from the SQLite database'''

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM messages_clean", engine)
    
    # define features and label arrays
    X = df['message']
    y = df.drop(columns=['message'])
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    '''Builds a text processing pipeline'''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [WordNetLemmatizer().lemmatize(w).strip() for w in tokens
              if w not in stopwords.words("english")] 

    return tokens


def build_model():
    '''Builds a machine learning pipeline'''

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
        ])
    # specify parameters for grid search
    parameters = {
        'clf__estimator__max_depth': [10, 20, 80],
        'clf__estimator__min_samples_leaf': [2, 4], 
        'clf__estimator__min_samples_split': [2, 10],
        'clf__estimator__n_estimators': [200]}

    # create grid search object
    model = GridSearchCV(pipeline, parameters)

    return model


def evaluate_model(model, X_test, y_test, category_names):
    '''Predict and Output results on the test set'''

    y_pred = model.predict(X_test)
    report = classification_report(y_test.values[:,], y_pred, target_names=category_names)
    return print(report)


def save_model(model, model_filepath):
    '''Exports the final model as a pickle file'''
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
        evaluate_model(model, X_test, y_test)

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