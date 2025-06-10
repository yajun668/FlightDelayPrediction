#This file is to generate initial delay score without spreading and ANSP Delay score and save results into a csv file

#import packages
import time,calendar
from datetime import datetime
from datetime import time as datetime_time
import networkx as nx
import pandas as pd
import numpy as np
import math
from sympy import * #use to calculate a variable value given a function

# generate a delay score and save result into a csv file
def generate_delay_score_csv_file(df, airport_list, date_range, my_alpha,my_beta,my_gamma,delay_threshold,hours_to_fly_threshold, outputname):
  df_delay_score = pd.DataFrame(columns=['Airport', 'Initial_z_score', 'Delay_Score','Datetime']) #store result and then save to csv file
  for date1 in date_range:
    my_date = date1.date()
    for time_hour in range(0,24):
      #print('Z score at time', time_hour,'---------')
      z = find_delay_vector(df, my_gamma, airport_list, delay_threshold, time_hour,my_date)
      #print("initial z score is:", z)
      W = find_Matrix_W(df, my_beta, airport_list, hours_to_fly_threshold, time_hour,my_date)
      #print('Matrix W: ', W)
      converged_delay_score = propagation(W, z, my_alpha)
      #print("after propagation, the z score is",converged_delay_score)
      # convert pd data series data into dataframe
      df_tmp = converged_delay_score.reset_index()
      df_tmp.columns = ['Airport', 'Delay_Score']
      #add z value to df_tmp
      df_tmp['Initial_z_score'] = df_tmp['Airport'].map(z)
      time_component = datetime_time(time_hour, 0)  # eg. 4:00 AM
      # Combine date and time
      combined_datetime = datetime.combine(my_date, time_component)
      # Add a new column "datetime" with the same datetime for all rows
      df_tmp['Datetime'] = combined_datetime
      # Append the DataFrame to the current DataFrame
      df_delay_score = pd.concat([df_delay_score, df_tmp], ignore_index=True)
  df_delay_score.to_csv(outputname, index = False)
  return df_delay_score # or delete to free up memory

# return each airport initial delay vector z at a specified time (hour) time_h and date_d
def find_delay_vector(df, my_gamma, airport_list, delay_threshold, time_hour,my_date):
  #Select flights based on their arrival times occurring within the last 3 hours relative to the time time_hour in date date_day
  #Also, we are only interested in flights with delay

  reference_datetime = pd.to_datetime(my_date) + pd.Timedelta(hours=time_hour)
  filtered_df = df.loc[(df['ARR_DEL15'] == 1) &
                       (df['Actual_ARR_dt_EST'] <= reference_datetime) &
                       (df['Actual_ARR_dt_EST'] >= reference_datetime - pd.Timedelta(hours=delay_threshold))
                       ]

  # calculate the time (hours) passed since the actual arrival time when the flight is labelled as delay

  # Calculate the time difference in hours
  filtered_df['elapsed_hours'] = reference_datetime - filtered_df['Actual_ARR_dt_EST']
  # Round up the values in the 'hour_difference' column to the nearest integer
  filtered_df['elapsed_hours'] = filtered_df['elapsed_hours'].apply(lambda x: math.ceil(x.total_seconds() / 3600))

  #print('filter dataframe is: ', filtered_df)
  # initialize vector z
  z = {key: 0 for key in airport_list} # delay vector: initialize 0 for each airport

  # calculate z for each airport in the Top 20 airport
  for airport in airport_list:
    tmp_df =  filtered_df.loc[filtered_df['DEST'] == airport] # find all flights whose destination is equal to airport
    if not tmp_df.empty:
       z[airport] = np.exp(-my_gamma*tmp_df['elapsed_hours']).sum()

  return z

# find normalized weight W matrix:
# an approach based on the during from the current time to next flight
def find_Matrix_W(df, my_beta, airport_list, hours_to_fly_threshold, time_hour,my_date):
  #For each pair of Airport flighting direction A-> B: select flights based on:
  # 1. Scheduled departure time from Airport A to  B occurring within the last hours_to_fly_threshold (.eg. 5 hours) relative to the time time_hour

  reference_datetime = pd.to_datetime(my_date) + pd.Timedelta(hours=time_hour)
  filtered_df = df.loc[(df['Scheduled_ARR_EST'] >= reference_datetime) &
                       (df['Scheduled_ARR_EST'] <= reference_datetime + pd.Timedelta(hours=hours_to_fly_threshold))
                       ]

  # calculate the the time (hours) to fly from A to B from the reference_datetime
  # Calculate the time difference in hours
  filtered_df['to_fly_hours'] = df['Scheduled_ARR_EST'] - reference_datetime
  # Round up the values in the 'hour_difference' column to the nearest integer
  filtered_df['to_fly_hours'] = filtered_df['to_fly_hours'].apply(lambda x: math.ceil(x.total_seconds() / 3600))

  # weight for delay matrix:  initialize 0 for each pair of airports
  W = {key1: {key2: 0 for key2 in airport_list} for key1 in airport_list}

  # iterate each pair of nodes
  num_nodes = len(airport_list)
  for idx1 in range(num_nodes):
    A = airport_list[idx1] # Airport A name
    for idx2 in range(num_nodes):
      B = airport_list[idx2] # Airport B name
      # Using .loc to filter based on two conditions
      df_A_B = filtered_df.loc[(filtered_df['ORIGIN'] == A) & (filtered_df['DEST'] == B)]
      if not df_A_B.empty:
        # find the minimum value of column filtered_df['to_fly_hours']: only need to find the earliest flight to B
        min_to_fly_hours = filtered_df['to_fly_hours'].min()
        W[A][B] = np.exp(-my_beta*min_to_fly_hours)

  matrix = np.array([list(inner_dict.values()) for inner_dict in W.values()])

  # Column-normalize the matrix
  column_sums = np.sum(matrix, axis=0)
  normalized_matrix = matrix / column_sums

  # Update the original dictionary with the normalized values
  for i, (key, inner_dict) in enumerate(W.items()):
      W[key] = {inner_key: normalized_matrix[i, j] for j, inner_key in enumerate(inner_dict.keys())}
  return W

#propagation function
def propagation(W,z,alpha):
  df_W = pd.DataFrame(W)
  df_W = df_W.fillna(0)  # Fill NaN values with 0 in the entire DataFrame
  df_z = pd.Series(z)

  epsilon = 0.001
  max_iteration = 10000
  iter_k = 0 #initialization
  diff = 10 #initialization

  prior_ksi = df_z.copy()
  current_ksi = df_z.copy()
  # if iteration reached max or diff<epsilon, the while loop terminates
  while iter_k < max_iteration and diff > epsilon:
    iter_k += 1
    current_ksi = alpha * np.dot(df_W,prior_ksi) + (1-alpha)*df_z
    vector_diff = current_ksi - prior_ksi
    # Find the largest absolute value in the vector
    diff = abs(max(vector_diff, key=abs))
    prior_ksi = current_ksi.copy()

  return current_ksi

  print('The total number of iterations required for convergence is ', iter_k)
  #print('The returned epsilon is ', diff)
  print('The returned convergence vector is ', current_ksi)