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
import dagshub 
from ingestion import x_train,x_test,y_train,y_test, params,
import hyperopt
from hyperopt import fmin, tpe,hp 
dagshub.init(repo_owner='Devrcb18', repo_name='fraud-detection-project-.mlflow', run_name='model_building')
mlflow.set_tracking_uri("https://dagshub.com/Devrcb18/fraud-detection-project-.mlflow")
mlflow.autolog()
mlflow.set_experiment("clickbait-classification")
with mlflow.start_run():
    model= XGBClassifier()
    
def objective_function(params):
    with mlflow.start_run():
      model = XGBClassifier(eval_metric='logloss', random_state=42)
      
    



