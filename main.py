# This is the main file to run the code for the whole project with two main steps

#Step 1: generate features of initial dealy score, ANSP delay score, and network centrality
#Step 2: Integrate these features with established features in literature, and run the empirical validation

#import packages
import os
import time,calendar
from datetime import datetime
from datetime import time as datetime_time
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt
from sympy import * #use to calulate a variable value given a function
import networkx as nx
import pandas as pd
#import python files
import helper,delay_score,network_feature
import prediction_model

#Read data and Hyperparameters Settings
# As the input_data.csv is around 2.3G, it is stored at mendeley and can be downloaded from

df_all_data = pd.read_csv("./data/input_data.csv")
#Large HUB airport list specified by FAA
selected_airport_list = ['ATL', 'AUS', 'BNA', 'BOS', 'BWI', 'CLT', 'DCA', 'DEN', 'DFW',
       'DTW', 'EWR', 'FLL', 'IAD', 'IAH', 'JFK', 'LAS', 'LAX', 'LGA',
       'MCO', 'MDW', 'MIA', 'MSP', 'ORD', 'PHL', 'PHX', 'SAN', 'SEA',
       'SFO', 'SLC', 'TPA']

#in this project we used dataset of June and July 2023
start_month = 6 # the start month is June 2023
end_month = 7
alpha = 0.85
beta = 2.3
gamma = 0.9
delay_threshold = 2
hours_to_fly_threshold = 5

# Records with both Origin and Destination are in Large HUB selected airport
df_selected_Airport = df_all_data[(df_all_data['ORIGIN'].isin(selected_airport_list)) & (df_all_data['DEST'].isin(selected_airport_list))]
print(df_selected_Airport.shape)
variables_selected = ['MONTH','ORIGIN','DEST','ARR_DEL15','DISTANCE','Scheduled_ARR_EST','Actual_ARR_dt_EST']
# # Data preprocessing
df_processed = helper.data_preprocess(df_selected_Airport, variables_selected)
print(df_processed.shape)

# step 1: generate initial dealy score, ANSP delay score, and network centrality for months June and July 2023
print('Step 1 is running...')
for month in range(start_month, end_month + 1):
    # dataframe for such month
    df_month = df_processed[df_processed['MONTH'] == month]
    print(f"the dataframe size in month {month} is {df_month.shape}")
    # generate network centrality feature
    output_folder_feature = './network_feature/'
    # if the folder doesn't exist, create the folder
    if not os.path.exists(output_folder_feature):
        os.makedirs(output_folder_feature)
    network_feature_filename = output_folder_feature + 'network_feature_month=' + str(month) + '.csv'
    frequency_df = network_feature.create_frequency_matrix(df_month, month, selected_airport_list)
    df_network_feature = network_feature.get_graph_features(frequency_df, network_feature_filename).T.rename_axis(
        'Airport').reset_index()

    # generate delay score
    # Create a datetime index for each day in month 2023
    date_range = helper.generate_month_date_range(2023, month)
    output_folder_delay_score = './delay_score/'
    # if the folder doesn't exist, create the folder
    if not os.path.exists(output_folder_delay_score):
        os.makedirs(output_folder_delay_score)
    csv_file_name = output_folder_delay_score + 'month=' + str(month) + 'alpha=' + str(
        alpha) + 'beta=' + str(beta) + 'gamma=' + str(gamma) + 'delay_score.csv'
    df_delay_score = delay_score.generate_delay_score_csv_file(df_month, selected_airport_list, date_range,
                                                               alpha, beta, gamma, delay_threshold,
                                                               hours_to_fly_threshold, csv_file_name)

    # plot the average delay_score across all airports for each hour from 0 to 23
    df_delay_score['Hour'] = df_delay_score['Datetime'].dt.hour
    # Group by 'Hour' and calculate average delay_score
    average_delay_hourly = df_delay_score.groupby('Hour')['Delay_Score'].mean()
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(average_delay_hourly, marker='o', linestyle='-')
    plot_title = 'month=' + str(month) + 'alpha=' + str(alpha) + 'beta=' + str(
        beta) + 'gamma =' + str(gamma) + 'Average Delay Score'
    plt.title(plot_title)
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Delay Score')
    plt.xticks(range(24))
    plt.grid(True)
    plt.tight_layout()

    output_folder_figure = './figure/'
    # if the folder doesn't exist, create the folder
    if not os.path.exists(output_folder_figure):
        os.makedirs(output_folder_figure)
    figure_title = output_folder_figure + 'month=' + str(month) + 'alpha=' + str(alpha) + 'beta=' + str(
        beta) + 'gamma =' + str(gamma) + '_averge_delay_score.pdf'
    plt.savefig(figure_title)
    # Close the plot
    plt.close()

# Step 2: Integrate these features with established features in literature, and run the empirical validation
# As empirical validation contains long code, we directly run the python file: empirical_validation.py
import empirical_validation 