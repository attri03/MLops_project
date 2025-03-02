import pandas as pd
import numpy as numpy
import os
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

log_file = 'logs'
os.makedirs(log_file, exist_ok= True)

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_file, "feature_engineering.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        logger.debug("File loaded successfully")
        return data
    except:
        logger.error("Exception raised during File loading")

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply TfIdf to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('tfidf applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise

def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame) -> None:
    try:
        dir_path = os.path.join('data', 'final')
        os.makedirs(dir_path)

        train_data_saving_path = os.path.join(dir_path, 'train_final.csv')
        train_data.to_csv(train_data_saving_path)

        test_data_saving_path = os.path.join(dir_path, 'test_final.csv')
        train_data.to_csv(test_data_saving_path)

        logger.debug('Data saved successfully')
    except:
        logger.error('Exception raised during saving of the file')

def main():
    try:
        max_features = 500

        #loading the training data
        train_file_path = os.path.join('data', 'interim', 'train_processed.csv')
        train_data = load_data(train_file_path)

        #loading the testing data
        test_file_path = os.path.join('data', 'interim', 'test_processed.csv')
        test_data = load_data(test_file_path)

        #feature engineering on the data
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)
        
        #Saving the data
        save_data(train_df, test_df)

        #logger
        logger.debug('Completed the feature engineering stage')
    except:
        logger.error('Exception raised during main')

if __name__ == "__main__":
    main()

