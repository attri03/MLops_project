import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_name = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_name)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_path_url: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path_url)
        logger.debug('Data loaded successfully')
        return data
    except:
        logger.error('Exception raised during loading the data')

def pre_processing(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data.drop(columns = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace = True)
        data.rename(columns = {'v1':'target', 'v2':'text'}, inplace = True)
        logger.debug('Pre-Processing of data completed successfully')
        return data
    except:
        logger.error('Exception raised during pre-processing of the data')

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        os.makedirs(data_path)
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path)
        train_data_path = os.path.join(raw_data_path, 'Train_data.csv')
        test_data_path = os.path.join(raw_data_path, 'Test_data.csv')
        train_data.to_csv(train_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)
    except:
        logger.error('Exception raised during saving the data')

def main():
    try:
        #parms
        data_path_url = "experiments\spam.csv"
        test_size = 0.2
        data_saving_path = 'data'
        #Loading the data
        data = load_data(data_path_url)
        #Pre-Processing data
        data = pre_processing(data)
        #train test split
        train_data, test_data = train_test_split(data, test_size = test_size, random_state = 2)
        #save data
        save_data(train_data, test_data, data_saving_path)
        #logging
        logger.debug('Data ingestion completed')
    except:
        logger.error('Exception raised in main function')

if __name__ == '__main__':
    main()