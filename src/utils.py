# Importing Neccessary Libraries
import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info("Exception Occured at save_object function")
        raise CustomException(e,sys)

## Example usecase of save_object:
# Incase of saving pre_processor as pkl we will call save_object function that will take two arguments
# 1. file_path = the path where we want to store pkl file inside a directory
# 2. obj = the object(pre_processor) we want to store in the given file_path
# It will read the directory path (as per Linux based system)
# It will make a directory on that path and will not make a new one if it already exists (exist_ok = True)
# It will open the file path in write byte mode
# It will read the file and store it as a pickle file (pickle.dump) at the given file_path

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]

            # Train Model
            model.fit(X_train,y_train)

            # Predict Training Data
            y_train_pred = model.predict(X_train)

            # Predict Testing Data
            y_test_pred = model.predict(X_test)

            # Get R2 scores for train & test data
            traim_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            return report

    except Exception as e:
        logging.info("Exception Occured while Evaluating Model")
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        logging.info("Exception Occured while loading object (utils)")
        raise CustomException(e,sys)