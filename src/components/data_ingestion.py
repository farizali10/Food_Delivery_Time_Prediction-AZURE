### IMPORTING NECESSARY LIBRARIES ###
import os # When deploying thru linux servers path is created through "os" only so we use to ensure smooth execution of the code
# So always use OS for saving/creating or perofrming any file function.

import sys # sys for getting system errors that would be included in exception handling
from src.logger import logging # Importing logging
from src.exception import CustomException # Importing exception
import pandas as pd 
from sklearn.model_selection import train_test_split # For train test split

## NEW FEATURE ##
from dataclasses import dataclass # No need to write __init__ for initializing variables. 
# Will work when we only want to create/store class variables and dont perform any functionalities inside the class.

## Initialise the Data Ingestion Configuration ##
# The main aim of data ingestion is to give paths as input and as output we get train,test data on those given paths.
# Due to which we are creating a separate dataclass which will have no more functionalities except paths. 

@dataclass # This is how we use dataclass
# Creating class that will contain all the required inputs for data_ingestion.py:
class DataIngestionConfig:
     train_data_path:str = os.path.join('artifacts','train.csv') # The path will look like this "artifacts/train.csv"
     test_data_path:str = os.path.join('artifacts','test.csv') # test_data_path:str (Not Compulsory but shown in documentation) is showing that the variable will contain str
     raw_data_path:str = os.path.join('artifacts','raw.csv')

# Creating a class for performing Data Ingestion:
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # This variable will contain test,train,raw paths. 
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Method Starts") 

        try:
            data = pd.read_csv(os.path.join("/config/workspace/notebooks/data","finalTrain.csv")) # We call utils.py here for reading a file from any source (MongoDB, SQL, etc)
            logging.info("Dataset read as pandas DataFrame")

            ## Removing unnecessary columns
            logging.info("Removing unnecessary columns")
            data = data.drop(labels=["ID"],axis=1)
            data = data.drop(labels=["Delivery_person_ID"],axis=1)
            data = data.drop(labels=["Time_Orderd"],axis=1)
            data = data.drop(labels=["Time_Order_picked"],axis=1)

            data["Order_Date"] = pd.to_datetime(data["Order_Date"],format='%d-%m-%Y')
            data["Day"] = pd.to_datetime(data["Order_Date"],format='%d-%m-%Y').dt.day
            data["Month"] = pd.to_datetime(data["Order_Date"],format='%d-%m-%Y').dt.month
            data["Year"] = pd.to_datetime(data["Order_Date"],format='%d-%m-%Y').dt.year
            data.drop("Order_Date",axis=1,inplace=True)

            data.rename(columns={"Day": "Day_Ordered", "Month": "Month_Ordered", "Year": "Year_Ordered", "Time_taken (min)":"Time_taken_min"}, inplace=True)
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) # exist_ok = True means that if raw data path already created dont create again
            data.to_csv(self.ingestion_config.raw_data_path,index=False) # Will save data to the given path
            # index = False is used when you dont want indexes of the DataFrame

            logging.info("Train Test Split")
            train_set, test_set = train_test_split(data,test_size=0.30,random_state=20)
            # We are using train_set, test_set instead of X_train, X_test, y_train, y_test as we have to save it

            # Saving train,test data in artifacts
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header = True) # headers = True will store column name
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header = True)

            logging.info("Data Ingestion Completed")

            ### OUTPUT ###
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage")
            raise CustomException(e,sys)

