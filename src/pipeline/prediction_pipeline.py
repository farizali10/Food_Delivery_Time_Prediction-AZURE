import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object,save_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred


        except Exception as e:
            logging.info("Exception Occured in Prediction Pipeline")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                Delivery_person_Age:float,
                Delivery_person_Ratings:float,
                Restaurant_latitude:float,
                Restaurant_longitude:float,
                Delivery_location_latitude:float,
                Delivery_location_longitude:float,
                Weather_conditions:object, 
                Road_traffic_density:object,
                Vehicle_condition:int,  
                Type_of_order:object, 
                Type_of_vehicle:object, 
                multiple_deliveries:float,
                Festival:object, 
                City:object,
                Day_Ordered:int,
                Month_Ordered:int,
                Year_Ordered:int):

        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Restaurant_latitude =  Restaurant_latitude
        self.Restaurant_longitude = Restaurant_longitude
        self.Delivery_location_latitude = Delivery_location_latitude
        self.Delivery_location_longitude = Delivery_location_longitude
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.Festival = Festival
        self.City = City
        self.Day_Ordered = Day_Ordered
        self.Month_Ordered = Month_Ordered
        self.Year_Ordered = Year_Ordered

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Delivery_person_Age":[self.Delivery_person_Age],
                "Delivery_person_Ratings":[self.Delivery_person_Ratings],
                "Restaurant_latitude":[self.Restaurant_latitude],
                "Restaurant_longitude":[self.Restaurant_longitude],
                "Delivery_location_latitude":[self.Delivery_location_latitude],
                "Delivery_location_longitude":[self.Delivery_location_longitude],
                "Weather_conditions":[self.Weather_conditions],
                "Road_traffic_density":[self.Road_traffic_density],
                "Vehicle_condition":[self.Vehicle_condition],
                "Type_of_order":[self.Type_of_order],
                "Type_of_vehicle":[self.Type_of_vehicle],
                "multiple_deliveries":[self.multiple_deliveries],
                "Festival":[self.Festival],
                "City":[self.City],
                "Day_Ordered":[self.Day_Ordered],
                "Month_Ordered":[self.Month_Ordered],
                "Year_Ordered":[self.Year_Ordered]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return df

        except Exception as e:
            logging.info("Exception Occured at get_data_as_dataframe (func)")
            raise CustomException(e,sys)