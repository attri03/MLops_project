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

def feature_engineering(train_data:pd.DataFrame, test_data:pd.DataFrame) -> pd.DataFrame:
    try:
        #Breaking the data into input and target columns
        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        #Applying tfid
        tfid = TfidfVectorizer(max_features = 500)
        X_train = tfid.fit_transform(X_train['text']).toarray()
        y_train = y_train['target'].values
        X_test = tfid.fit_transform(X_test['text']).toarray()
        y_test = y_test['target'].values
        
        train_df = pd.DataFrame(X_train.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test.toarray())
        test_df['label'] = y_test

        logger.debug("Completed feature engineering stage")
        return train_df, test_df
    except:
        pass

def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame) -> pd.DataFrame:
    try:
        pass
    except:
        pass

def main():
    try:
        pass
    except:
        pass

if __name__ == "__main__":
    main()

