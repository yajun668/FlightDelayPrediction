# -*- coding: utf-8 -*-
"""Empirical validation.ipynb
# Airline project empirical validation
"""
print('Step 2: running empirical validation...')

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, auc, roc_auc_score, confusion_matrix, f1_score
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint, EarlyStopping
import itertools

month = 6
alpha = 0.85
beta = 2.3
gamma = 0.9

#df_all_data.ORIGIN.max()

# data merge and preprocessing
df_all_data = pd.read_csv("./data/input_data.csv")
Top30AP = np.unique(df_all_data[df_all_data.FAA_class == "Lrg"].ORIGIN)

df = df_all_data.loc[(df_all_data.MONTH == month) | (df_all_data.MONTH == month+1)]
df = df[(df['ORIGIN'].isin(Top30AP)) & (df['DEST'].isin(Top30AP))]
df['Scheduled_DEP_EST_adj'] = df['Scheduled_DEP_EST'].astype('datetime64[h]')

net_var1 = pd.read_csv("./data/network_features/network_feature_month="+str(month)+".csv")
net_var1['month'] = month
net_var2 = pd.read_csv("./data/network_features/network_feature_month="+str(month+1)+".csv")
net_var2['month'] = month + 1
net_var = pd.concat([net_var1, net_var2], axis=0)

delay_score1 = pd.read_csv("./data/delay_score/month="+str(month)+"alpha="+str(alpha)+"beta="+str(beta)+"gamma="+str(gamma)+"delay_score.csv")
delay_score2 = pd.read_csv("./data/delay_score/month="+str(month+1)+"alpha="+str(alpha)+"beta="+str(beta)+"gamma="+str(gamma)+"delay_score.csv")
delay_score = pd.concat([delay_score1, delay_score2], axis=0)
delay_score['post2_Datetime'] = pd.to_datetime(delay_score['Datetime']) + timedelta(hours=2)
delay_score['post3_Datetime'] = pd.to_datetime(delay_score['Datetime']) + timedelta(hours=3)
delay_score['post6_Datetime'] = pd.to_datetime(delay_score['Datetime']) + timedelta(hours=6)

df_new = df.merge(net_var, left_on = ["ORIGIN","MONTH"], right_on = ["Airport","month"], how = 'left')
df_new = df_new.merge(delay_score[["Airport","Delay_Score", "Initial_z_score","post2_Datetime"]], left_on = ["ORIGIN","Scheduled_DEP_EST_adj"], right_on = ["Airport","post2_Datetime"], how = 'left')
df_new = df_new.rename(columns = {"Delay_Score": "Delay_Score_2hr", "Initial_z_score": "Initial_z_score_2hr"})
df_new = df_new.merge(delay_score[["Airport","Delay_Score", "Initial_z_score","post3_Datetime"]], left_on = ["ORIGIN","Scheduled_DEP_EST_adj"], right_on = ["Airport","post3_Datetime"], how = 'left')
df_new = df_new.rename(columns = {"Delay_Score": "Delay_Score_3hr", "Initial_z_score": "Initial_z_score_3hr"})
df_new = df_new.merge(delay_score[["Airport","Delay_Score", "Initial_z_score","post6_Datetime"]], left_on = ["ORIGIN","Scheduled_DEP_EST_adj"], right_on = ["Airport","post6_Datetime"], how = 'left')
df_new = df_new.rename(columns = {"Delay_Score": "Delay_Score_6hr", "Initial_z_score": "Initial_z_score_6hr"})

Carrier = pd.get_dummies(df_new[["OP_CARRIER"]]).iloc[:, :-1]
day_of_week = pd.get_dummies(df_new[["day_of_week"]]).iloc[:, :-1]
ORIGIN = pd.get_dummies(df_new[["ORIGIN"]]).iloc[:, :-1]
df_new = pd.concat([df_new, Carrier, day_of_week, ORIGIN], axis=1).drop(['OP_CARRIER', 'day_of_week', 'ORIGIN'], axis=1)

df_new = df_new.rename(columns = {"Arr_1hrpre_num": "Arr_1hr_Prior_Num", "Arr_1hrpost_num": "Arr_1hr_Post_Num",
                                 "DEP_1hrpre_num": "Dep_1hr_Prior_Num", "DEP_1hrpost_num": "Dep_1hr_Post_Num",
                                 "max_temp_f": "Max_Temp_F", "min_temp_f": "Min_Temp_F", "avg_wind_speed_kts": "Avg_Wind_Speed_Kts",
                                 "precip_in": "Precip_In", "affected_turnaround_lessthan60": "Scheduled_Turnaround_Lessthan60",
                                 "betweenness_centrality": "Betweenness_Centrality", "closeness_centrality": "Closeness_Centrality",
                                 "Delay_Score_2hr": "ANSP_Score"})

# define 3 sets of features. variables_baseline is from the airline operation literature,
# variables_centrality is about the network centrality, variables_delay_index is about the airport delay index.
variables_dependent = ['DEP_DEL15']
variables_baseline = ['Arr_1hr_Prior_Num', 'Arr_1hr_Post_Num', 'Dep_1hr_Prior_Num', 'Dep_1hr_Post_Num', 'Max_Temp_F',
                      'Min_Temp_F', 'Avg_Wind_Speed_Kts', 'Precip_In', 'Scheduled_Turnaround_Lessthan60'] + df_new.columns[
                        df_new.columns.str.startswith('OP_CARRIER')].tolist() + df_new.columns[
                        df_new.columns.str.startswith('day_of_week')].tolist() + df_new.columns[
                        df_new.columns.str.startswith('ORIGIN')].tolist()
variables_centrality = ['Betweenness_Centrality', 'Closeness_Centrality']
variables_delay_index = ['ANSP_Score', 'Delay_Score_3hr', 'Delay_Score_6hr']
variables_delay_initial_index = ["Initial_z_score_2hr", "Initial_z_score_3hr", "Initial_z_score_6hr"]
flight_information = ["FL_DATE", "DEST", "Scheduled_DEP", "Scheduled_ARR_Local", "Actual_ARR_dt_Local", "Actual_DEP_dt_EST", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]

reg_data = df_new[variables_dependent + variables_baseline + variables_centrality + variables_delay_index + variables_delay_initial_index + flight_information].dropna()

reg_data[reg_data.DEP_DEL15 == 1][['Arr_1hr_Prior_Num', 'Arr_1hr_Post_Num', 'Dep_1hr_Prior_Num', 'Dep_1hr_Post_Num', 'Max_Temp_F',
       'Min_Temp_F', 'Avg_Wind_Speed_Kts', 'Precip_In', 'Scheduled_Turnaround_Lessthan60',
      'Betweenness_Centrality', 'Closeness_Centrality', 'ANSP_Score']].describe().to_excel("./output/summary stats.xlsx")

"""# One hidden layer ANN"""

# Cross validation for ANN
cv = KFold(n_splits=5, random_state=0, shuffle=True)
x = reg_data.iloc[:,1:]
y = reg_data[variables_dependent]
p_pred1 = []
p_pred2 = []
p_pred3 = []
p_pred3_init = []
p_pred4 = []
p_pred4_init = []
p_pred5 = []
p_pred5_init = []
y_true  = []

for train_index, test_index in cv.split(reg_data):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    y_true = y_true + y_test.DEP_DEL15.tolist()

    class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = np.ravel(y_train))
    class_weights = dict(zip(np.unique(y_train), class_weights))
    sc = StandardScaler()
    x_train = pd.DataFrame(sc.fit_transform(x_train), columns = x.columns)
    x_test = pd.DataFrame(sc.transform(x_test), columns = x.columns)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience = 20)
    x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.20, random_state = 0)

# one hidden layer ANN

    ann2 = Sequential()
    ann2.add(Dense(np.round((len(x_train[variables_baseline + variables_centrality].columns)+1)*2/3), activation='relu', input_shape=(len(x_train[variables_baseline + variables_centrality].columns),)))
    ann2.add(Dense(1, activation='sigmoid'))
    ann2.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
    ann2.fit(x_tr[variables_baseline + variables_centrality], y_tr, validation_data=(x_va[variables_baseline + variables_centrality], y_va), epochs=1000, verbose=0, callbacks=[es], class_weight = class_weights)
    p = ann2.predict(x_test[variables_baseline + variables_centrality])
    p_pred2 = p_pred2 + p.tolist()

    ann3 = Sequential()
    ann3.add(Dense(np.round((len(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]].columns)+1)*2/3), activation='relu', input_shape=(len(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]].columns),)))
    ann3.add(Dense(1, activation='sigmoid'))
    ann3.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
    ann3.fit(x_tr[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_tr, validation_data=(x_va[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_va), epochs=1000, verbose=0, callbacks=[es], class_weight = class_weights)
    p = ann3.predict(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
    p_pred3 = p_pred3 + p.tolist()

    ann3_init = Sequential()
    ann3_init.add(Dense(np.round((len(x_train[variables_baseline + variables_centrality + variables_delay_initial_index[0:1]].columns)+1)*2/3), activation='relu', input_shape=(len(x_train[variables_baseline + variables_centrality + variables_delay_initial_index[0:1]].columns),)))
    ann3_init.add(Dense(1, activation='sigmoid'))
    ann3_init.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
    ann3_init.fit(x_tr[variables_baseline + variables_centrality + variables_delay_initial_index[0:1]], y_tr, validation_data=(x_va[variables_baseline + variables_centrality + variables_delay_initial_index[0:1]], y_va), epochs=1000, verbose=0, callbacks=[es], class_weight = class_weights)
    p = ann3_init.predict(x_test[variables_baseline + variables_centrality + variables_delay_initial_index[0:1]])
    p_pred3_init = p_pred3_init + p.tolist()

    ann4 = Sequential()
    ann4.add(Dense(np.round((len(x_train[variables_baseline + variables_centrality + variables_delay_index[1:2]].columns)+1)*2/3), activation='relu', input_shape=(len(x_train[variables_baseline + variables_centrality + variables_delay_index[1:2]].columns),)))
    ann4.add(Dense(1, activation='sigmoid'))
    ann4.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
    ann4.fit(x_tr[variables_baseline + variables_centrality + variables_delay_index[1:2]], y_tr, validation_data=(x_va[variables_baseline + variables_centrality + variables_delay_index[1:2]], y_va), epochs=1000, verbose=0, callbacks=[es], class_weight = class_weights)
    p = ann4.predict(x_test[variables_baseline + variables_centrality + variables_delay_index[1:2]])
    p_pred4 = p_pred4 + p.tolist()

    ann4_init = Sequential()
    ann4_init.add(Dense(np.round((len(x_train[variables_baseline + variables_centrality + variables_delay_initial_index[1:2]].columns)+1)*2/3), activation='relu', input_shape=(len(x_train[variables_baseline + variables_centrality + variables_delay_initial_index[1:2]].columns),)))
    ann4_init.add(Dense(1, activation='sigmoid'))
    ann4_init.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
    ann4_init.fit(x_tr[variables_baseline + variables_centrality + variables_delay_initial_index[1:2]], y_tr, validation_data=(x_va[variables_baseline + variables_centrality + variables_delay_initial_index[1:2]], y_va), epochs=1000, verbose=0, callbacks=[es], class_weight = class_weights)
    p = ann4_init.predict(x_test[variables_baseline + variables_centrality + variables_delay_initial_index[1:2]])
    p_pred4_init = p_pred4_init + p.tolist()

    ann5 = Sequential()
    ann5.add(Dense(np.round((len(x_train[variables_baseline + variables_centrality + variables_delay_index[2:3]].columns)+1)*2/3), activation='relu', input_shape=(len(x_train[variables_baseline + variables_centrality + variables_delay_index[2:3]].columns),)))
    ann5.add(Dense(1, activation='sigmoid'))
    ann5.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
    ann5.fit(x_tr[variables_baseline + variables_centrality + variables_delay_index[2:3]], y_tr, validation_data=(x_va[variables_baseline + variables_centrality + variables_delay_index[2:3]], y_va), epochs=1000, verbose=0, callbacks=[es], class_weight = class_weights)
    p = ann5.predict(x_test[variables_baseline + variables_centrality + variables_delay_index[2:3]])
    p_pred5 = p_pred5 + p.tolist()

    ann5_init = Sequential()
    ann5_init.add(Dense(np.round((len(x_train[variables_baseline + variables_centrality + variables_delay_initial_index[2:3]].columns)+1)*2/3), activation='relu', input_shape=(len(x_train[variables_baseline + variables_centrality + variables_delay_initial_index[2:3]].columns),)))
    ann5_init.add(Dense(1, activation='sigmoid'))
    ann5_init.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
    ann5_init.fit(x_tr[variables_baseline + variables_centrality + variables_delay_initial_index[2:3]], y_tr, validation_data=(x_va[variables_baseline + variables_centrality + variables_delay_initial_index[2:3]], y_va), epochs=1000, verbose=0, callbacks=[es], class_weight = class_weights)
    p = ann5_init.predict(x_test[variables_baseline + variables_centrality + variables_delay_initial_index[2:3]])
    p_pred5_init = p_pred5_init + p.tolist()

p_pred2 = list(itertools.chain.from_iterable(p_pred2))
p_pred3 = list(itertools.chain.from_iterable(p_pred3))
p_pred3_init = list(itertools.chain.from_iterable(p_pred3_init))
p_pred4 = list(itertools.chain.from_iterable(p_pred4))
p_pred4_init = list(itertools.chain.from_iterable(p_pred4_init))
p_pred5 = list(itertools.chain.from_iterable(p_pred5))
p_pred5_init = list(itertools.chain.from_iterable(p_pred5_init))

# report AUC score

print("Results for one hidden layer ANN:")
print("")
print("AUC for baseline features + network centrality features:", roc_auc_score(y_true, p_pred2))
print("AUC for baseline features + network centrality features + 2-hours lag delay index:", roc_auc_score(y_true, p_pred3))
print("AUC for baseline features + network centrality features + 2-hours lag z_init index", roc_auc_score(y_true, p_pred3_init))
print("AUC for baseline features + network centrality features + 3-hours lag delay index:", roc_auc_score(y_true, p_pred4))
print("AUC for baseline features + network centrality features + 3-hours lag z_init index:", roc_auc_score(y_true, p_pred4_init))
print("AUC for baseline features + network centrality features + 6-hours lag delay index:", roc_auc_score(y_true, p_pred5))
print("AUC for baseline features + network centrality features + 6-hours lag z_init index:", roc_auc_score(y_true, p_pred5_init))
print("")
print("F1 score for baseline features + network centrality features:", f1_score(y_true, np.round(np.array(p_pred2))))
print("F1 score for baseline features + network centrality features + 2-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred3))))
print("F1 score for baseline features + network centrality features + 2-hours lag z_init index:", f1_score(y_true, np.round(np.array(p_pred3_init))))
print("F1 score for baseline features + network centrality features + 3-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred4))))
print("F1 score for baseline features + network centrality features + 3-hours lag z_init index:", f1_score(y_true, np.round(np.array(p_pred4_init))))
print("F1 score for baseline features + network centrality features + 6-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred5))))
print("F1 score for baseline features + network centrality features + 6-hours lag z_init index:", f1_score(y_true, np.round(np.array(p_pred5_init))))

# report sensitivity while controlling specificity at 0.8 level.


threshold2=0.5810194
y_pred2 = np.zeros(len(p_pred2))
y_pred2[np.array(p_pred2) > threshold2] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred2).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold3=0.5879665
y_pred3 = np.zeros(len(p_pred3))
y_pred3[np.array(p_pred3) > threshold3] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred3).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features + 2-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold3_init=0.5856174
y_pred3_init = np.zeros(len(p_pred3_init))
y_pred3_init[np.array(p_pred3_init) > threshold3_init] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred3_init).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + 2-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold4=0.5786092
y_pred4 = np.zeros(len(p_pred4))
y_pred4[np.array(p_pred4) > threshold4] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred4).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features + 3-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold4_init=0.5775255
y_pred4_init = np.zeros(len(p_pred4_init))
y_pred4_init[np.array(p_pred4_init) > threshold4_init] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred4_init).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + 3-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold5=0.5880263
y_pred5 = np.zeros(len(p_pred5))
y_pred5[np.array(p_pred5) > threshold5] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred5).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features + 6-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold5_init=0.603313
y_pred5_init = np.zeros(len(p_pred5_init))
y_pred5_init[np.array(p_pred5_init) > threshold5_init] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred5_init).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + 6-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

"""# Random forest"""

# Cross validation for random forest
cv = KFold(n_splits=5, random_state=0, shuffle=True)
x = reg_data.iloc[:,1:]
y = reg_data[variables_dependent]
p_pred1 = []
p_pred2 = []
p_pred3 = []
p_pred3_init = []
p_pred4 = []
p_pred4_init = []
p_pred5 = []
p_pred5_init = []
y_true  = []

for train_index, test_index in cv.split(reg_data):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    y_true = y_true + y_test.DEP_DEL15.tolist()

# Random Forest

    rf2 = RandomForestClassifier(n_estimators = 200, random_state = 0, max_features = "sqrt", class_weight='balanced_subsample').fit(x_train[variables_baseline + variables_centrality], y_train.values.ravel())
    p = rf2.predict_proba(x_test[variables_baseline + variables_centrality])
    p_pred2 = p_pred2 + p[:,1].tolist()

    rf3 = RandomForestClassifier(n_estimators = 200, random_state = 0, max_features = "sqrt", class_weight='balanced_subsample').fit(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_train.values.ravel())
    p = rf3.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
    p_pred3 = p_pred3 + p[:,1].tolist()

    rf3_init = RandomForestClassifier(n_estimators=200, random_state=0, max_features="sqrt",
                                      class_weight='balanced_subsample').fit(
        x_train[variables_baseline + variables_centrality + variables_delay_initial_index[0:1]], y_train.values.ravel())
    p = rf3_init.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_initial_index[0:1]])
    p_pred3_init = p_pred3_init + p[:, 1].tolist()

    rf4 = RandomForestClassifier(n_estimators = 200, random_state = 0, max_features = "sqrt", class_weight='balanced_subsample').fit(x_train[variables_baseline + variables_centrality + variables_delay_index[1:2]], y_train.values.ravel())
    p = rf4.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[1:2]])
    p_pred4 = p_pred4 + p[:,1].tolist()

    rf4_init = RandomForestClassifier(n_estimators=200, random_state=0, max_features="sqrt",
                                      class_weight='balanced_subsample').fit(
        x_train[variables_baseline + variables_centrality + variables_delay_initial_index[1:2]], y_train.values.ravel())
    p = rf4_init.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_initial_index[1:2]])
    p_pred4_init = p_pred4_init + p[:, 1].tolist()

    rf5 = RandomForestClassifier(n_estimators = 200, random_state = 0, max_features = "sqrt", class_weight='balanced_subsample').fit(x_train[variables_baseline + variables_centrality + variables_delay_index[2:3]], y_train.values.ravel())
    p = rf5.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[2:3]])
    p_pred5 = p_pred5 + p[:,1].tolist()

    rf5_init = RandomForestClassifier(n_estimators=200, random_state=0, max_features="sqrt",
                                      class_weight='balanced_subsample').fit(
        x_train[variables_baseline + variables_centrality + variables_delay_initial_index[2:3]], y_train.values.ravel())
    p = rf5_init.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_initial_index[2:3]])
    p_pred5_init = p_pred5_init + p[:, 1].tolist()

# report AUC score

print("Results for one random forest:")
print("")
print("AUC for baseline features + network centrality features:", roc_auc_score(y_true, p_pred2))
print("AUC for baseline features + network centrality features + 2-hours lag delay index:", roc_auc_score(y_true, p_pred3))
print("AUC for baseline features + network centrality features + 2-hours lag z_init index", roc_auc_score(y_true, p_pred3_init))
print("AUC for baseline features + network centrality features + 3-hours lag delay index:", roc_auc_score(y_true, p_pred4))
print("AUC for baseline features + network centrality features + 3-hours lag z_init index:", roc_auc_score(y_true, p_pred4_init))
print("AUC for baseline features + network centrality features + 6-hours lag delay index:", roc_auc_score(y_true, p_pred5))
print("AUC for baseline features + network centrality features + 6-hours lag z_init index:", roc_auc_score(y_true, p_pred5_init))
print("")
print("F1 score for baseline features + network centrality features:", f1_score(y_true, np.round(np.array(p_pred2))))
print("F1 score for baseline features + network centrality features + 2-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred3))))
print("F1 score for baseline features + network centrality features + 2-hours lag z_init index:", f1_score(y_true, np.round(np.array(p_pred3_init))))
print("F1 score for baseline features + network centrality features + 3-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred4))))
print("F1 score for baseline features + network centrality features + 3-hours lag z_init index:", f1_score(y_true, np.round(np.array(p_pred4_init))))
print("F1 score for baseline features + network centrality features + 6-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred5))))
print("F1 score for baseline features + network centrality features + 6-hours lag z_init index:", f1_score(y_true, np.round(np.array(p_pred5_init))))

# report sensitivity while controlling specificity at 0.8 level.

threshold2=0.415
y_pred2 = np.zeros(len(p_pred2))
y_pred2[np.array(p_pred2) > threshold2] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred2).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold3=0.41
y_pred3 = np.zeros(len(p_pred3))
y_pred3[np.array(p_pred3) > threshold3] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred3).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features + 2-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold3_init=0.41
y_pred3_init = np.zeros(len(p_pred3_init))
y_pred3_init[np.array(p_pred3_init) > threshold3_init] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred3_init).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + 2-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold4=0.41
y_pred4 = np.zeros(len(p_pred4))
y_pred4[np.array(p_pred4) > threshold4] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred4).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features + 3-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold4_init=0.405
y_pred4_init = np.zeros(len(p_pred4_init))
y_pred4_init[np.array(p_pred4_init) > threshold4_init] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred4_init).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + 3-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold5=0.41
y_pred5 = np.zeros(len(p_pred5))
y_pred5[np.array(p_pred5) > threshold5] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred5).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features + 6-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold5_init=0.41
y_pred5_init = np.zeros(len(p_pred5_init))
y_pred5_init[np.array(p_pred5_init) > threshold5_init] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred5_init).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + 6-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

"""# Bagging logistic regression"""

# Cross validation for bagged logistic regression with cost-sensitive learning
cv = KFold(n_splits=5, random_state=0, shuffle=True)
x = reg_data.iloc[:,1:]
y = reg_data[variables_dependent]
p_pred1 = []
p_pred2 = []
p_pred3 = []
p_pred3_init = []
p_pred4 = []
p_pred4_init = []
p_pred5 = []
p_pred5_init = []
y_true  = []

for train_index, test_index in cv.split(reg_data):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    y_true = y_true + y_test.DEP_DEL15.tolist()

# Logistic regression

    lr2 = BaggingClassifier(base_estimator = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'), n_estimators = 200, random_state = 0).fit(x_train[variables_baseline + variables_centrality], y_train.values.ravel())
    p = lr2.predict_proba(x_test[variables_baseline + variables_centrality])
    p_pred2 = p_pred2 + p[:,1].tolist()

    lr3 = BaggingClassifier(base_estimator = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'), n_estimators = 200, random_state = 0).fit(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_train.values.ravel())
    p = lr3.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
    p_pred3 = p_pred3 + p[:,1].tolist()

    lr3_init = BaggingClassifier(base_estimator = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'), n_estimators = 200, random_state = 0).fit(x_train[variables_baseline + variables_centrality + variables_delay_initial_index[0:1]], y_train.values.ravel())
    p = lr3_init.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_initial_index[0:1]])
    p_pred3_init = p_pred3_init + p[:,1].tolist()

    lr4 = BaggingClassifier(base_estimator = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'), n_estimators = 200, random_state = 0).fit(x_train[variables_baseline + variables_centrality + variables_delay_index[1:2]], y_train.values.ravel())
    p = lr4.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[1:2]])
    p_pred4 = p_pred4 + p[:,1].tolist()

    lr4_init = BaggingClassifier(base_estimator = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'), n_estimators = 200, random_state = 0).fit(x_train[variables_baseline + variables_centrality + variables_delay_initial_index[1:2]], y_train.values.ravel())
    p = lr4_init.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_initial_index[1:2]])
    p_pred4_init = p_pred4_init + p[:,1].tolist()

    lr5 = BaggingClassifier(base_estimator = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'), n_estimators = 200, random_state = 0).fit(x_train[variables_baseline + variables_centrality + variables_delay_index[2:3]], y_train.values.ravel())
    p = lr5.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[2:3]])
    p_pred5 = p_pred5 + p[:,1].tolist()

    lr5_init = BaggingClassifier(base_estimator = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'), n_estimators = 200, random_state = 0).fit(x_train[variables_baseline + variables_centrality + variables_delay_initial_index[2:3]], y_train.values.ravel())
    p = lr5_init.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_initial_index[2:3]])
    p_pred5_init = p_pred5_init + p[:,1].tolist()

# report AUC score

print("Results for bagged cost-sensitive logistic regression:")
print("")
print("AUC for baseline features + network centrality features:", roc_auc_score(y_true, p_pred2))
print("AUC for baseline features + network centrality features + 2-hours lag delay index:", roc_auc_score(y_true, p_pred3))
print("AUC for baseline features + network centrality features + 2-hours lag z_init index", roc_auc_score(y_true, p_pred3_init))
print("AUC for baseline features + network centrality features + 3-hours lag delay index:", roc_auc_score(y_true, p_pred4))
print("AUC for baseline features + network centrality features + 3-hours lag z_init index:", roc_auc_score(y_true, p_pred4_init))
print("AUC for baseline features + network centrality features + 6-hours lag delay index:", roc_auc_score(y_true, p_pred5))
print("AUC for baseline features + network centrality features + 6-hours lag z_init index:", roc_auc_score(y_true, p_pred5_init))
print("")
print("F1 score for baseline features + network centrality features:", f1_score(y_true, np.round(np.array(p_pred2))))
print("F1 score for baseline features + network centrality features + 2-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred3))))
print("F1 score for baseline features + network centrality features + 2-hours lag z_init index:", f1_score(y_true, np.round(np.array(p_pred3_init))))
print("F1 score for baseline features + network centrality features + 3-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred4))))
print("F1 score for baseline features + network centrality features + 3-hours lag z_init index:", f1_score(y_true, np.round(np.array(p_pred4_init))))
print("F1 score for baseline features + network centrality features + 6-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred5))))
print("F1 score for baseline features + network centrality features + 6-hours lag z_init index:", f1_score(y_true, np.round(np.array(p_pred5_init))))

# report sensitivity while controlling specificity at 0.8 level.

threshold2=0.5782106
y_pred2 = np.zeros(len(p_pred2))
y_pred2[np.array(p_pred2) > threshold2] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred2).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold3=0.5798106
y_pred3 = np.zeros(len(p_pred3))
y_pred3[np.array(p_pred3) > threshold3] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred3).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features + 2-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold3_init=0.5679349
y_pred3_init = np.zeros(len(p_pred3_init))
y_pred3_init[np.array(p_pred3_init) > threshold3_init] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred3_init).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + 2-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold4=0.5781453
y_pred4 = np.zeros(len(p_pred4))
y_pred4[np.array(p_pred4) > threshold4] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred4).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features + 3-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold4_init=0.5700567
y_pred4_init = np.zeros(len(p_pred4_init))
y_pred4_init[np.array(p_pred4_init) > threshold4_init] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred4_init).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + 3-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold5=0.5767345
y_pred5 = np.zeros(len(p_pred5))
y_pred5[np.array(p_pred5) > threshold5] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred5).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features + 6-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold5_init=0.5803937
y_pred5_init = np.zeros(len(p_pred5_init))
y_pred5_init[np.array(p_pred5_init) > threshold5_init] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred5_init).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + 6-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

"""# XGBoost"""

# Cross validation for XGBoost with cost sensitive learning
cv = KFold(n_splits=5, random_state=0, shuffle=True)
x = reg_data.iloc[:,1:]
y = reg_data[variables_dependent]
p_pred1 = []
p_pred2 = []
p_pred3 = []
p_pred3_init = []
p_pred4 = []
p_pred4_init = []
p_pred5 = []
p_pred5_init = []
y_true  = []

for train_index, test_index in cv.split(reg_data):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    y_true = y_true + y_test.DEP_DEL15.tolist()

# XGBoost

    xg2 = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(x_train[variables_baseline + variables_centrality], y_train.values.ravel())
    p = xg2.predict_proba(x_test[variables_baseline + variables_centrality])
    p_pred2 = p_pred2 + p[:,1].tolist()

    xg3 = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_train.values.ravel())
    p = xg3.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
    p_pred3 = p_pred3 + p[:,1].tolist()

    xg3_init = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(x_train[variables_baseline + variables_centrality + variables_delay_initial_index[0:1]], y_train.values.ravel())
    p = xg3_init.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_initial_index[0:1]])
    p_pred3_init = p_pred3_init + p[:,1].tolist()

    xg4 = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(x_train[variables_baseline + variables_centrality + variables_delay_index[1:2]], y_train.values.ravel())
    p = xg4.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[1:2]])
    p_pred4 = p_pred4 + p[:,1].tolist()

    xg4_init = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(x_train[variables_baseline + variables_centrality + variables_delay_initial_index[1:2]], y_train.values.ravel())
    p = xg4_init.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_initial_index[1:2]])
    p_pred4_init = p_pred4_init + p[:,1].tolist()

    xg5 = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(x_train[variables_baseline + variables_centrality + variables_delay_index[2:3]], y_train.values.ravel())
    p = xg5.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[2:3]])
    p_pred5 = p_pred5 + p[:,1].tolist()

    xg5_init = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(x_train[variables_baseline + variables_centrality + variables_delay_initial_index[2:3]], y_train.values.ravel())
    p = xg5_init.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_initial_index[2:3]])
    p_pred5_init = p_pred5_init + p[:,1].tolist()

# report AUC score and F1 score

print("Results for XGboost:")
print("")
print("AUC for baseline features + network centrality features:", roc_auc_score(y_true, p_pred2))
print("AUC for baseline features + network centrality features + 2-hours lag delay index:", roc_auc_score(y_true, p_pred3))
print("AUC for baseline features + network centrality features + 2-hours lag z_init index", roc_auc_score(y_true, p_pred3_init))
print("AUC for baseline features + network centrality features + 3-hours lag delay index:", roc_auc_score(y_true, p_pred4))
print("AUC for baseline features + network centrality features + 3-hours lag z_init index:", roc_auc_score(y_true, p_pred4_init))
print("AUC for baseline features + network centrality features + 6-hours lag delay index:", roc_auc_score(y_true, p_pred5))
print("AUC for baseline features + network centrality features + 6-hours lag z_init index:", roc_auc_score(y_true, p_pred5_init))
print("")
print("F1 score for baseline features + network centrality features:", f1_score(y_true, np.round(np.array(p_pred2))))
print("F1 score for baseline features + network centrality features + 2-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred3))))
print("F1 score for baseline features + network centrality features + 2-hours lag z_init index:", f1_score(y_true, np.round(np.array(p_pred3_init))))
print("F1 score for baseline features + network centrality features + 3-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred4))))
print("F1 score for baseline features + network centrality features + 3-hours lag z_init index:", f1_score(y_true, np.round(np.array(p_pred4_init))))
print("F1 score for baseline features + network centrality features + 6-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred5))))
print("F1 score for baseline features + network centrality features + 6-hours lag z_init index:", f1_score(y_true, np.round(np.array(p_pred5_init))))

# report sensitivity while controlling specificity at 0.8 level.

threshold2=0.5750394
y_pred2 = np.zeros(len(p_pred2))
y_pred2[np.array(p_pred2) > threshold2] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred2).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold3=0.5739571
y_pred3 = np.zeros(len(p_pred3))
y_pred3[np.array(p_pred3) > threshold3] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred3).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features + 2-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold3_init=0.5718996
y_pred3_init = np.zeros(len(p_pred3_init))
y_pred3_init[np.array(p_pred3_init) > threshold3_init] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred3_init).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + 2-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold4=0.5733084
y_pred4 = np.zeros(len(p_pred4))
y_pred4[np.array(p_pred4) > threshold4] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred4).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features + 3-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold4_init=0.5692106
y_pred4_init = np.zeros(len(p_pred4_init))
y_pred4_init[np.array(p_pred4_init) > threshold4_init] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred4_init).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + 3-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold5=0.5738746
y_pred5 = np.zeros(len(p_pred5))
y_pred5[np.array(p_pred5) > threshold5] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred5).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features + 6-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold5_init=0.5744269
y_pred5_init = np.zeros(len(p_pred5_init))
y_pred5_init[np.array(p_pred5_init) > threshold5_init] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred5_init).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + 6-hours lag delay index from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

"""# XGBoost based feature importance"""

import xgboost

xg = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(reg_data[variables_baseline + variables_centrality + variables_delay_index[1:2]], reg_data[variables_dependent].values.ravel())

ax = xgboost.plot_importance(xg)
fig = ax.figure
fig.set_size_inches(20, 10)

"""# Comparison of four models on 2-hours lag delay index"""

# Cross validation for comparison of three machine learning models
cv = KFold(n_splits=5, random_state=0, shuffle=True)
x = reg_data.iloc[:,1:]
y = reg_data[variables_dependent]
p_pred_ann = []
p_pred_rf = []
p_pred_lr = []
p_pred_xg = []
y_true  = []

for train_index, test_index in cv.split(reg_data):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    y_true = y_true + y_test.DEP_DEL15.tolist()

# rerun one hidden layer ANN, random forest, bagged logistic regression, and xgboost

    rf3 = RandomForestClassifier(n_estimators = 200, random_state = 0, max_features = "sqrt", class_weight='balanced_subsample').fit(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_train.values.ravel())
    p = rf3.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
    p_pred_rf = p_pred_rf + p[:,1].tolist()

    lr3 = BaggingClassifier(base_estimator = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'), n_estimators = 200, random_state = 0).fit(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_train.values.ravel())
    p = lr3.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
    p_pred_lr = p_pred_lr + p[:,1].tolist()

    xg3 = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_train.values.ravel())
    p = xg3.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
    p_pred_xg = p_pred_xg + p[:,1].tolist()

    class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = np.ravel(y_train))
    class_weights = dict(zip(np.unique(y_train), class_weights))
    sc = StandardScaler()
    x_train = pd.DataFrame(sc.fit_transform(x_train), columns = x.columns)
    x_test = pd.DataFrame(sc.transform(x_test), columns = x.columns)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience = 20)
    x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.20, random_state = 0)

    ann3 = Sequential()
    ann3.add(Dense(np.round((len(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]].columns)+1)*2/3), activation='relu', input_shape=(len(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]].columns),)))
    ann3.add(Dense(1, activation='sigmoid'))
    ann3.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
    ann3.fit(x_tr[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_tr, validation_data=(x_va[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_va), epochs=1000, verbose=0, callbacks=[es], class_weight = class_weights)
    p = ann3.predict(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
    p_pred_ann = p_pred_ann + p.tolist()

from sklearn.metrics import roc_curve

fpr_ann , tpr_ann, thresholds_ann = roc_curve(y_true, p_pred_ann)
fpr_rf , tpr_rf, thresholds_rf = roc_curve(y_true, p_pred_rf)
fpr_lr , tpr_lr, thresholds_lr = roc_curve(y_true, p_pred_lr)
fpr_xg , tpr_xg, thresholds_xg = roc_curve(y_true, p_pred_xg)

plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr_xg, tpr_xg, label= "XGBoost: AUC = 0.7456")
plt.plot(fpr_rf, tpr_rf, label= "Random forest: AUC = 0.7243")
plt.plot(fpr_lr, tpr_lr, label= "Bagged logistic: AUC = 0.7036")
plt.plot(fpr_ann, tpr_ann, label= "ANN: AUC = 0.7287")
plt.legend()
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title('Receiver Operating Characteristic')
plt.savefig("./output/rrr.pdf", format="pdf")
plt.show()

"""# Out-of-time validation"""

reg_data = df_new[variables_dependent + variables_baseline + variables_centrality + variables_delay_index + variables_delay_initial_index + ["MONTH"]].dropna()
train = reg_data[reg_data["MONTH"] == 6]
test = reg_data[reg_data["MONTH"] == 7]
x_train = train.iloc[:,1:-1]
y_train = train[variables_dependent]
x_test = test.iloc[:,1:-1]
y_test = test[variables_dependent]

y_true = y_test.DEP_DEL15.tolist()

rf2 = RandomForestClassifier(n_estimators = 200, random_state = 0, max_features = "sqrt", class_weight='balanced_subsample').fit(x_train[variables_baseline + variables_centrality], y_train.values.ravel())
p = rf2.predict_proba(x_test[variables_baseline + variables_centrality])
p_pred_rf0 = p[:,1].tolist()

rf3 = RandomForestClassifier(n_estimators = 200, random_state = 0, max_features = "sqrt", class_weight='balanced_subsample').fit(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_train.values.ravel())
p = rf3.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
p_pred_rf = p[:,1].tolist()

lr2 = BaggingClassifier(base_estimator = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'), n_estimators = 200, random_state = 0).fit(x_train[variables_baseline + variables_centrality], y_train.values.ravel())
p = lr2.predict_proba(x_test[variables_baseline + variables_centrality])
p_pred_lr0 = p[:,1].tolist()

lr3 = BaggingClassifier(base_estimator = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'), n_estimators = 200, random_state = 0).fit(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_train.values.ravel())
p = lr3.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
p_pred_lr = p[:,1].tolist()

xg2 = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(x_train[variables_baseline + variables_centrality], y_train.values.ravel())
p = xg2.predict_proba(x_test[variables_baseline + variables_centrality])
p_pred_xg0 = p[:,1].tolist()

xg3 = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_train.values.ravel())
p = xg3.predict_proba(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
p_pred_xg = p[:,1].tolist()

class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = np.ravel(y_train))
class_weights = dict(zip(np.unique(y_train), class_weights))
sc = StandardScaler()
x_train = pd.DataFrame(sc.fit_transform(x_train), columns = x_train.columns)
x_test = pd.DataFrame(sc.transform(x_test), columns = x_train.columns)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience = 20)
x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.20, random_state = 0)

ann2 = Sequential()
ann2.add(Dense(np.round((len(x_train[variables_baseline + variables_centrality].columns)+1)*2/3), activation='relu', input_shape=(len(x_train[variables_baseline + variables_centrality].columns),)))
ann2.add(Dense(1, activation='sigmoid'))
ann2.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
ann2.fit(x_tr[variables_baseline + variables_centrality], y_tr, validation_data=(x_va[variables_baseline + variables_centrality], y_va), epochs=1000, verbose=0, callbacks=[es], class_weight = class_weights)
p = ann2.predict(x_test[variables_baseline + variables_centrality])
p_pred_ann0 = p.tolist()

ann3 = Sequential()
ann3.add(Dense(np.round((len(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]].columns)+1)*2/3), activation='relu', input_shape=(len(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]].columns),)))
ann3.add(Dense(1, activation='sigmoid'))
ann3.compile(optimizer='adam', loss="binary_crossentropy",metrics=['accuracy'])
ann3.fit(x_tr[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_tr, validation_data=(x_va[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_va), epochs=1000, verbose=0, callbacks=[es], class_weight = class_weights)
p = ann3.predict(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
p_pred_ann = p.tolist()

# report AUC score and F1 score

print("Results for XGboost:")
print("")
print("AUC for baseline features + network centrality features:", roc_auc_score(y_true, p_pred_xg0))
print("AUC for baseline features + network centrality features + 2-hours lag delay index:", roc_auc_score(y_true, p_pred_xg))
print("")
print("F1 score for baseline features + network centrality features:", f1_score(y_true, np.round(np.array(p_pred_xg0))))
print("F1 score for baseline features + network centrality features + 2-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred_xg))))

print("Results for Random forest:")
print("")
print("AUC for baseline features + network centrality features:", roc_auc_score(y_true, p_pred_rf0))
print("AUC for baseline features + network centrality features + 2-hours lag delay index:", roc_auc_score(y_true, p_pred_rf))
print("")
print("F1 score for baseline features + network centrality features:", f1_score(y_true, np.round(np.array(p_pred_rf0))))
print("F1 score for baseline features + network centrality features + 2-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred_rf))))

print("Results for Random logistic regression:")
print("")
print("AUC for baseline features + network centrality features:", roc_auc_score(y_true, p_pred_lr0))
print("AUC for baseline features + network centrality features + 2-hours lag delay index:", roc_auc_score(y_true, p_pred_lr))
print("")
print("F1 score for baseline features + network centrality features:", f1_score(y_true, np.round(np.array(p_pred_lr0))))
print("F1 score for baseline features + network centrality features + 2-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred_lr))))

print("Results for ANN:")
print("")
print("AUC for baseline features + network centrality features:", roc_auc_score(y_true, p_pred_ann0))
print("AUC for baseline features + network centrality features + 2-hours lag delay index:", roc_auc_score(y_true, p_pred_ann))
print("")
print("F1 score for baseline features + network centrality features:", f1_score(y_true, np.round(np.array(p_pred_ann0))))
print("F1 score for baseline features + network centrality features + 2-hours lag delay index:", f1_score(y_true, np.round(np.array(p_pred_ann))))

# report sensitivity while controlling specificity at 0.8 level.

threshold_xg0=0.627144
y_pred_xg0 = np.zeros(len(p_pred_xg0))
y_pred_xg0[np.array(p_pred_xg0) > threshold_xg0] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_xg0).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold_xg=0.630556
y_pred_xg = np.zeros(len(p_pred_xg))
y_pred_xg[np.array(p_pred_xg) > threshold_xg] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_xg).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold_rf0=0.405
y_pred_rf0 = np.zeros(len(p_pred_rf0))
y_pred_rf0[np.array(p_pred_rf0) > threshold_rf0] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_rf0).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold_rf=0.415
y_pred_rf = np.zeros(len(p_pred_rf))
y_pred_rf[np.array(p_pred_rf) > threshold_rf] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_rf).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold_lr0=0.6307946
y_pred_lr0 = np.zeros(len(p_pred_lr0))
y_pred_lr0[np.array(p_pred_lr0) > threshold_lr0] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_lr0).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold_lr=0.629421
y_pred_lr = np.zeros(len(p_pred_lr))
y_pred_lr[np.array(p_pred_lr) > threshold_lr] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_lr).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold_ann0=0.6987191
y_pred_ann0 = np.zeros(len(p_pred_ann0))
y_pred_ann0[np.array(p_pred_ann0) > threshold_ann0] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_ann0).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

threshold_ann=0.7009194
y_pred_ann = np.zeros(len(p_pred_ann))
y_pred_ann[np.array(p_pred_ann) > threshold_ann] = 1
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_ann).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)
print("Sensitivity for baseline features + network centrality features from literature:", sensitivity)
print("Specificity for baseline features + network centrality features from literature:", specificity)

"""# SHAP analysis"""

import shap
reg_data = df_new[variables_dependent + variables_baseline + variables_centrality + variables_delay_index + variables_delay_initial_index + ["MONTH"]].dropna()
train = reg_data[reg_data["MONTH"] == 6]
test = reg_data[reg_data["MONTH"] == 7]
x_train = train.iloc[:,1:-1]
y_train = train[variables_dependent]
x_test = test.iloc[:,1:-1]
y_test = test[variables_dependent]

xg = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_train.values.ravel())

explainer = shap.Explainer(xg.predict, x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]])
shap_values = explainer(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])
shap.plots.bar(shap_values, max_display = 5)

reg_data = df_new[variables_dependent + variables_baseline + variables_centrality + variables_delay_index + variables_delay_initial_index + ["MONTH"]].dropna()
train = reg_data[reg_data["MONTH"] == 6]
test = reg_data[reg_data["MONTH"] == 7]
x_train = train.iloc[:,1:-1]
y_train = train[variables_dependent]
x_test = test.iloc[:,1:-1]
y_test = test[variables_dependent]

xg = XGBClassifier(n_estimators = 200, random_state = 0, scale_pos_weight = 1/np.mean(y_train.values)-1).fit(x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_train.values.ravel())

reg_data = df_new[variables_dependent + variables_baseline + variables_centrality + variables_delay_index + variables_delay_initial_index + flight_information + ["MONTH"]].dropna()
train = reg_data[reg_data["MONTH"] == 6]
test = reg_data[reg_data["MONTH"] == 7]
x_train = train.iloc[:,1:-1]
y_train = train[variables_dependent]
x_test = test.iloc[:,1:-1]
y_test = test[variables_dependent]

test.iloc[6,:].to_csv("./output/shap1.csv")

df_new.iloc[346859,:][variables_dependent + variables_baseline + variables_centrality + variables_delay_index + variables_delay_initial_index + flight_information + ["MONTH"]].to_csv("./output/shap2.csv")

test.iloc[[20680,20698,20728,20895,21993,23234,23571,24391,28984,32848],][variables_dependent + variables_baseline + variables_centrality + variables_delay_index + variables_delay_initial_index + ["MONTH"] + flight_information].to_csv("./output/shap.csv")

explainer1 = shap.TreeExplainer(xg, x_train[variables_baseline + variables_centrality + variables_delay_index[0:1]])
shap_values1 = explainer1.shap_values(x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]])

y_test.iloc[171143,:]

test.iloc[[11,171143],].to_csv("./output/shap.csv")

shap.summary_plot(
    shap_values1, features=x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], feature_names=variables_baseline + variables_centrality + variables_delay_index[0:1], plot_type="bar", max_display=5, show=False)
plt.savefig('./output/shap1.pdf', format='pdf', dpi=600, bbox_inches='tight')

shap.force_plot(explainer1.expected_value, shap_values1[171143], feature_names = explainer1.data_feature_names, link = 'logit')

shap.save_html('./output/shap4b.html', shap.force_plot(explainer1.expected_value, shap_values1[171143], feature_names=explainer1.data_feature_names, link='logit'))

shap.initjs()

shap.force_plot(shap_values1[16152], matplotlib = True, show = False, figsize = (30,4), link = 'logit')

shap.initjs()
shap.force_plot(shap_values[10002], matplotlib = True, show = False , figsize = (28,4))

plt.savefig('./output/shap4b.pdf', format='pdf', dpi=600, bbox_inches='tight')

shap.plots.bar(shap_values[:,(0, 2, 5, 7, 65)],show=False)
plt.savefig('./output/shap1.pdf', format='pdf', dpi=600, bbox_inches='tight')

shap.summary_plot(shap_values1, x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], max_display=5, show=False)
plt.savefig('./output/shap2.pdf', format='pdf', dpi=600, bbox_inches='tight')

shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="DEP_DEL15",show=False)
plt.savefig('./output/shap3.pdf', format='pdf', dpi=600, bbox_inches='tight')

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 12))
axes = axes.ravel()
shap.dependence_plot("Delay_Score_2", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="DEP_DEL15", ax=axes[0], show=False)
axes[0].set(xlabel = None)
shap.dependence_plot("Delay_Score_2", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="Arr_1hrpre_num", ax=axes[1], show=False)
axes[1].set(xlabel = None)
axes[1].set(ylabel = None)
shap.dependence_plot("Delay_Score_2", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="Arr_1hrpost_num", ax=axes[2], show=False)
axes[2].set(xlabel = None)
axes[2].set(ylabel = None)
shap.dependence_plot("Delay_Score_2", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="DEP_1hrpre_num", ax=axes[3], show=False)
axes[3].set(xlabel = None)
axes[3].set(ylabel = None)
shap.dependence_plot("Delay_Score_2", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="DEP_1hrpost_num", ax=axes[4], show=False)
axes[4].set(xlabel = None)
shap.dependence_plot("Delay_Score_2", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="max_temp_f", ax=axes[5], show=False)
axes[5].set(xlabel = None)
axes[5].set(ylabel = None)
shap.dependence_plot("Delay_Score_2", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="min_temp_f", ax=axes[6], show=False)
axes[6].set(xlabel = None)
axes[6].set(ylabel = None)
shap.dependence_plot("Delay_Score_2", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="avg_wind_speed_kts", ax=axes[7], show=False)
axes[7].set(xlabel = None)
axes[7].set(ylabel = None)
shap.dependence_plot("Delay_Score_2", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="precip_in", ax=axes[8], show=False)
shap.dependence_plot("Delay_Score_2", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="affected_turnaround_lessthan60", ax=axes[9], show=False)
axes[9].set(ylabel = None)
shap.dependence_plot("Delay_Score_2", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="betweenness_centrality", ax=axes[10], show=False)
axes[10].set(ylabel = None)
shap.dependence_plot("Delay_Score_2", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="closeness_centrality", ax=axes[11], show=False)
axes[11].set(ylabel = None)

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 12))
axes = axes.ravel()
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_9E", ax=axes[0], show=False)
axes[0].set(xlabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_AA", ax=axes[1], show=False)
axes[1].set(xlabel = None)
axes[1].set(ylabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_AS", ax=axes[2], show=False)
axes[2].set(xlabel = None)
axes[2].set(ylabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_B6", ax=axes[3], show=False)
axes[3].set(xlabel = None)
axes[3].set(ylabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_C5", ax=axes[4], show=False)
axes[4].set(xlabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_DL", ax=axes[5], show=False)
axes[5].set(xlabel = None)
axes[5].set(ylabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_F9", ax=axes[6], show=False)
axes[6].set(xlabel = None)
axes[6].set(ylabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_G4", ax=axes[7], show=False)
axes[7].set(xlabel = None)
axes[7].set(ylabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_G7", ax=axes[8], show=False)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_MQ", ax=axes[9], show=False)
axes[9].set(ylabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_NK", ax=axes[10], show=False)
axes[10].set(ylabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_OH", ax=axes[11], show=False)
axes[11].set(ylabel = None)

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 12))
axes = axes.ravel()
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_OO", ax=axes[0], show=False)
axes[0].set(xlabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_PT", ax=axes[1], show=False)
axes[1].set(xlabel = None)
axes[1].set(ylabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_QX", ax=axes[2], show=False)
axes[2].set(xlabel = None)
axes[2].set(ylabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_UA", ax=axes[3], show=False)
axes[3].set(xlabel = None)
axes[3].set(ylabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_WN", ax=axes[4], show=False)
axes[4].set(xlabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_YV", ax=axes[5], show=False)
axes[5].set(xlabel = None)
axes[5].set(ylabel = None)
shap.dependence_plot("Delay_Score_2hr", np.hstack([shap_values.values, np.zeros((len(shap_values.values), 1))]), pd.concat([x_test[variables_baseline + variables_centrality + variables_delay_index[0:1]], y_test], axis = 1), interaction_index="OP_CARRIER_YX", ax=axes[6], show=False)
axes[6].set(xlabel = None)
axes[6].set(ylabel = None)

reg_data.columns