import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
import dagshub 
import logging
import os 
import hyperopt
from hyperopt import fmin, tpe,hp 
import yaml

#------------------------logging configuration------------------------
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
#------------------------- Processed Data Loading-------------------------
def load_splits():
  try:
    """load preprocessed data"""
    base_path ='data/'
    x_train = np.load(base_path+"processed/x_train.npy", allow_pickle=True)
    x_test  = np.load(base_path+"processed/x_test.npy",  allow_pickle=True)
    y_train = np.load(base_path+"processed/y_train.npy", allow_pickle=True)
    y_test  = np.load(base_path+"processed/y_test.npy",  allow_pickle=True)
    return x_train, x_test, y_train, y_test
    logger.debug('the processed data is fetched from %s',base_path)
  except Exception as E:
      logger.exception(f'error occured while fetching {E}') 

#------------------------parameter loading------------------------------
def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
# ---------------------- Dagshub and MLFLOW ----------------------------    
try:
        dagshub.init(repo_owner='Devrcb18', repo_name='fraud-detection-project-.mlflow', run_name='model_building')
        mlflow.set_tracking_uri("https://dagshub.com/Devrcb18/fraud-detection-project-.mlflow")
        logger.debug('dagshub and mlflow repo initialised')   
except Exception as e:
    logger.error(f"while initialising following error occurred : {e}")          
mlflow.autolog(disable=True)
mlflow.set_experiment("clickbait-classification")

# ------------------------------ Hyperopt optimisation using TPE------------------    
def objective_function(params):
    x_train, x_test, y_train, y_test =load_splits()
    model_params = {
        "max_depth": int(params["max_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "n_estimators": int(params["n_estimators"]),
        "eval_metric": "logloss",
        "random_state": 42,
        "use_label_encoder": False,
    }
    with mlflow.start_run(nested=True):
        model = XGBClassifier(**model_params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        return {"loss": -acc}
    
if __name__ =='__main__':
        params = load_params("params.yaml")  
        max_depth_upper_limit = params['model_building']['max_depth_upper_limit']
        max_depth_lower_limit = params['model_building']['max_depth_lower_limit']
        learning_rate_upper_limit = params['model_building']['learning_rate_upper_limit']
        learning_rate_lower_limit = params['model_building']['learning_rate_lower_limit']
        num_estimator_choices = params['model_building']['num_estimator_choices']
        eval_metric = params['model_building']['eval_metric'] 
        random_state = params['model_building']['random_state']
             
        best= fmin(
            fn=objective_function,
                    space= {
                        'max_depth':hp.uniform('max_depth',max_depth_lower_limit,max_depth_upper_limit),
                        'learning_rate':hp.uniform('learning_rate',learning_rate_lower_limit,learning_rate_upper_limit),
                        'num_estimator':hp.choice('num_estimator',num_estimator_choices)
                        }, 
                    algo=tpe.suggest,
                    max_evals=30
        )
        logger.info(f"Best hyperparameters found: {best}")
        best_params = {
            'max_depth': int(best['max_depth']),
            'learning_rate': best['learning_rate'],
            'n_estimators': num_estimator_choices[best['num_estimator']],
            'eval_metric': 'logloss',
            'random_state': 42,
            'use_label_encoder': False
        }  
        x_train, x_test, y_train, y_test = load_splits()
        final_model = XGBClassifier(**best_params)
        final_model.fit(x_train, y_train)  
        final_model.fit(x_train, y_train)
        y_pred = final_model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        


