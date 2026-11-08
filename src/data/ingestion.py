import os 
import logging
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def data_entry(data_path:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.debug('Data loaded from %s', data_path)
        return df
    except FileNotFoundError:
        logger.error('File not found: %s', data_path)
        raise
    
def find_param(param_path:str)->dict:
    try:
        with open(param_path,'r') as f:
            params = yaml.load(f)
            logger.debug('Parameters loaded from %s', param_path)
            return params
    except FileNotFoundError:
        logger.error('Parameter file not found: %s', param_path)
        raise       
def data_split(data:pd.DataFrame)->pd.DataFrame:
    try:
        params = find_param('params.yaml')
        test_size = params['data_ingestion']['test_size']
        x = data['title']
        y = data['clickbait']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        logger.debug('Data split into train and test sets with test size %s', test_size)
        return x_train, x_test, y_train, y_test
    except Exception as E:
        logger.exception('Raised an exception %s',E)
        raise
     
def preprocess_data(x_train:pd.DataFrame,x_test:pd.DataFrame):
    """ this is function for preprocessing the title into vectors """
    try:
        params = find_param('params.yaml')
        test_size = params['feature_engineering']['max_features']
        vectorizer = TfidfVectorizer(max_features = 45)
        x_test_precrossed = vectorizer.fit_transform(x_test)
        x_train_precrossed = vectorizer.transform(x_train)
        return x_train_precrossed, x_test_precrossed
    except Exception as E:
        logger.exception('Raised an exception %s',E)
        raise
def save_data(x_train, x_test, y_train, y_test):
    try:
        np.save('x_train.npy', x_train)
        np.save('x_test.npy', x_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        logger.debug('Data saved to .npy files')
    except Exception as E:
        logger.exception('Raised an exception %s',E)
        raise
def main():
    try:
        params = find_param('params.yaml')
        data_path = params['data_ingestion']['file_path']
        data = data_entry(data_path)
        x_train, x_test, y_train, y_test = data_split(data)
        x_train_precrossed, x_test_precrossed = preprocess_data(x_train, x_test)
        save_data(x_train_precrossed, x_test_precrossed, y_train, y_test)
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")
    
if '__name__' == '__main__':
    main()    
    
    
    
    
       






























