### Importing Necessary Libraries ### 
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info("Data Trasnformation Initiated")
            
            # Seggregating Categorical and Numerical Features
            logging.info("Segregating Categorical and Numerical Features")

            ## Automated Way
            #categorical_columns = X.select_dtypes(include="object").columns
            #numerical_columns = X.select_dtypes(exclude="object").columns
            #datetime_columns = X.select_dtypes(include="datetime64[ns]").columns
            
            ## Manual Way
            numerical_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'Vehicle_condition', 'multiple_deliveries', 'Day_Ordered', 'Month_Ordered', 'Year_Ordered']
            cat_ordinal_columns = ["Weather_conditions","Road_traffic_density","City"] # Columns to be Ordinally Encoded
            cat_ohe_columns = ["Type_of_order","Type_of_vehicle","Festival"] # Columns to be One Hot Encoded
            
            Type_of_order_cat = ['Buffet', 'Drinks', 'Meal', 'Snack']
            Type_of_vehicle_cat = ['bicycle', 'electric_scooter', 'motorcycle', 'scooter']
            Festival_cat = ['No', 'Yes', 'nan']

            # Defining custom rankings for each Ordinal feature
            logging.info("Defining custom rankings for each Ordinal feature")

            Weather_conditions_cat = ['Sunny', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Fog', 'nan'] 
            Road_traffic_density_cat = ['Low', 'Medium', 'High', 'Jam', 'nan'] 
            City_cat = ['Urban', 'Metropolitian', 'Semi-Urban', 'nan']

            ### PIPELINES ###
            logging.info("Pipeline Initiated")

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")), # We chose median incase of presence of outliers.
                    ("scaler_numerical",StandardScaler(with_mean=False)) # Standardization + Min Max Scaler
                ])
            
            # Ordinal Encoding Pipeline
            cat_pipeline_ordinal = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder(categories= [Weather_conditions_cat,Road_traffic_density_cat,City_cat])),
                ('scaler_ordinal', StandardScaler(with_mean=False))
                ]

            )

            # One Hot Encoding Pipeline
            cat_pipeline_ohe = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(categories = [Type_of_order_cat,Type_of_vehicle_cat,Festival_cat])),
                ('scaler_ohe', StandardScaler(with_mean=False))
                ]
                )

            # Combining pipelines for both numerical as well as categorical features using column transformer.
            pre_processor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline_ordinal", cat_pipeline_ordinal, cat_ordinal_columns),
                ("cat_pipeline_ohe", cat_pipeline_ohe, cat_ohe_columns)
            ], remainder='passthrough')

            # With remainder = "passthrough" it makes sure that if client had given columns which are not be pre-processed so those columns wont be changed.
            
            logging.info("Pipeline Completed")

            return pre_processor

        except Exception as e:
            logging.info("Error in Data Trasnformation Stage")
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading Train and Test Data Completed")
            logging.info(f"Train DataFrame Head: \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head: \n{test_df.head().to_string()}")
            
            logging.info("Obtaining Pre-Processing Object")
            preprocessing_obj = self.get_data_transformation_object()
            
            target_column_name = 'Time_taken_min'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Transforming Train & Test Data using Pre-Processor Object
            logging.info("Applying Pre-Processing Object on Train and Test DataFrames")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Saving them so that we dont have to be perfroming all the pre_processing again for future use of data
            # Concatenating train/test_arr and target_feature_train/test_df and coverting into a numpy array as its easier to load and apply ML Algorithms 
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            # Concatenating input and target features as in future we would also need to perform train_test_split
            
            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path, obj = preprocessing_obj)
            logging.info("Pre-Processing PICKLE file saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception Occured at Initiating Data Transformation")
            raise CustomException(e,sys)



