# this file contains common functions used for other python files

#import packages
import time,calendar
from datetime import datetime
from datetime import time as datetime_time
import networkx as nx
import pandas as pd
import numpy as np
import math
from sympy import * #use to calulate a variable value given a function

# generate a date range for any given month and year;    
def generate_month_date_range(year, month):
    # Generate date range for the specified month and year
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}'
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    return date_range    

#𝜆 is calculated based on the ratio between air time (ACTUAL_ELAPSED_TIME) and distance. Specifically,
# we find the largest ratio of t and d, (t/d), based on each pair of airports (empirical data).
# 𝜆 is determined such that w_{AB} is close to one with the largest ratio of t and d.
def calculate_lambda(df):
  x = symbols('x') # assign x as a variable
  df['ratio'] = df['ACTUAL_ELAPSED_TIME'] / df['DISTANCE']
  lartest_ratio = df['ratio'].max()
  #print('The largest ratio is: ',lartest_ratio)
  expression = exp(lartest_ratio * x) / (1 + exp(lartest_ratio * x))
  my_lambda = solve(expression - 0.9999, x)
  return float(my_lambda[0])

# data file preprocess: return dataframe with interested features
def data_preprocess(df, variables_selected):
  new_df = df[variables_selected]
  # Drop rows with NaN values
  new_df = new_df.dropna()
  #conversion to datetime type
  new_df['Scheduled_ARR_EST'] = pd.to_datetime(new_df['Scheduled_ARR_EST'])
  new_df['Actual_ARR_dt_EST'] = pd.to_datetime(new_df['Actual_ARR_dt_EST'])
  new_df['MONTH'] = new_df['MONTH'].astype(int)  # convert into inteter
  new_df['ARR_DEL15'] = new_df['ARR_DEL15'].astype(int) #convert into inteter
  new_df['Actual_ARR_Hour'] = new_df['Actual_ARR_dt_EST'].dt.hour
  return new_df

# download a sharable file from google drive
def download_files_from_GoogleDrive(Shareablelink, outputfile):
  # Install PyDrive
  #!pip install pydrive2 # install pydrive if not already installed
  #Import modules
  from pydrive2.auth import GoogleAuth
  from pydrive2.drive import GoogleDrive
  from google.colab import auth
  from oauth2client.client import GoogleCredentials
  #Authenticate and create the PyDrive2 client
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)
  #Get the Shareable link
  #Ex link : https://drive.google.com/file/d/1-POAW0964JvYefW1q07D1cEBfc4hiJHj/view?usp=sharing
  # Get the id from the link 1c7Ffo1Go1dtUpKcSWxdbdVyW4dfhEoUp
  file_id = Shareablelink.split('/')[-2]
  downloaded = drive.CreateFile({'id':file_id})
  #downloaded = drive.CreateFile({'id':"1-POAW0964JvYefW1q07D1cEBfc4hiJHj"})
  downloaded.GetContentFile(outputfile)