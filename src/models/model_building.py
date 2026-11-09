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
from scipy.sparse import load_npz
import dagshub 
import logging
import os 
import hyperopt
from hyperopt import fmin, tpe,hp
from hyperopt import STATUS_OK 
import yaml
import joblib

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
    base_path = "data_files/processed/"
    x_train = load_npz(base_path + "x_train.npz")
    x_test = load_npz(base_path + "x_test.npz")
    y_train = np.load(base_path + "y_train.npy")
    y_test = np.load(base_path + "y_test.npy")
    logger.debug('the processed data is fetched from %s',base_path)
    return x_train.tocsr(), x_test.tocsr(), y_train.ravel(), y_test.ravel()
  except Exception as E:
      logger.exception(f'error occured while fetching {E}') 

#------------------------parameter loading------------------------------
def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
# ---------------------- Dagshub and MLFLOW ----------------------------    
try:
    dagshub.init(repo_owner="Devrcb18",repo_name="fraud-detection-project-",mlflow=True)
    logger.info("dagshub initialized successfully")
except Exception as E:
    logger.exception(f"dagshub initialization failed : {E}")
    raise
print("MLflow Tracking URI:", mlflow.get_tracking_uri())
mlflow.set_experiment("clickbait-classification")
mlflow.autolog(disable=True)

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
        return {"loss": -acc,"status": STATUS_OK}
    
#--------------------- Model Training and Tuning --------------------------    

def main():
    params = load_params("params.yaml")
    max_depth_upper_limit = params['model_building']['max_depth_upper_limit']
    max_depth_lower_limit = params['model_building']['max_depth_lower_limit']
    learning_rate_upper_limit = params['model_building']['learning_rate_upper_limit']
    learning_rate_lower_limit = params['model_building']['learning_rate_lower_limit']
    num_estimator_choices = params['model_building']['num_estimator_choices']
    eval_metric = params['model_building']['eval_metric']
    random_state = params['model_building']['random_state']
    assert learning_rate_lower_limit < learning_rate_upper_limit
    assert max_depth_lower_limit < max_depth_upper_limit
    logger.info("Hyperopt optimization starting")

    space = {
        'max_depth': hp.quniform('max_depth',max_depth_lower_limit,max_depth_upper_limit,1),
        'learning_rate': hp.loguniform('learning_rate',np.log(learning_rate_lower_limit),np.log(learning_rate_upper_limit)),
        'n_estimators': hp.choice('n_estimators',num_estimator_choices)
    }
    with mlflow.start_run(run_name="Clickbait_XGBoost"):
        best = fmin(
            fn=objective_function,
            space=space,
            algo=tpe.suggest,
            max_evals=30,
            show_progressbar=True
        )
        best_params = {
            'max_depth': int(best['max_depth']),
            'learning_rate': float(best['learning_rate']),
            'n_estimators': num_estimator_choices[best['n_estimators']],
            'eval_metric': eval_metric,
            'random_state': random_state
        }
        logger.info("Best hyperparameters: %s", best_params)
        x_train, x_test, y_train, y_test = load_splits()
        final_model = XGBClassifier(**best_params)
        final_model.fit(x_train, y_train)
        y_pred = final_model.predict(x_test)
        final_acc = accuracy_score(y_test, y_pred)
        final_f1 = f1_score(y_test, y_pred)
        mlflow.log_params(best_params)
        mlflow.log_metric("final_accuracy", final_acc)
        mlflow.log_metric("final_f1", final_f1)
        os.makedirs("models", exist_ok=True)
        joblib.dump(final_model,"models/clickbait_xgb.joblib")
        mlflow.log_artifact("models/clickbait_xgb.joblib")
        logger.info("accuracy: %.4f | f1: %.4f",final_acc,final_f1)
    logger.info("Hyperopt optimization completed")
    
if __name__== '__main__':
    main()    
    