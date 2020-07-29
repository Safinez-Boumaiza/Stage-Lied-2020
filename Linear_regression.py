#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Safinez BOUMAIZA
purpose : Compare the sensibility of  both polynomial UTCI and UTCI of Broed  regarding the
 different parameters
method : Multiple linear regression
"""

# Import packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

def data_management(input_file1,input_file2 ):
    '''
    This function aims to gather two dataframes into one based on the datetime column, calculate tmrt-Ta
    :param input_file: name of the data file to import
    :return: One input file (dataframe)
    '''
    # Import Data
    final_data = pd.read_excel(input_file1, indexcol=0, nrows= 10000)
    utci_tmrt = pd.read_excel(input_file2, indexcol=0, nrows = 10000)
    utci_tmrt.rename(columns={utci_tmrt.columns[0]: "datetime"}, inplace=True)

    # Jointure
    df_joined_inputs = pd.merge(left=final_data, right=utci_tmrt, left_on='datetime', right_on='datetime')

    # TMRT-Ta calculation
    df_joined_inputs['tmrt-ta'] = df_joined_inputs['T_mrt'] - df_joined_inputs['Temp_air_2m']

    # Creation of the table_offset
    df_interest_inputs = df_joined_inputs[
        ['datetime', 'Temp_air_2m', 'tmrt-ta', 'WindSpeed', 'RelativeHumidity', 'UTCI']]

    ## Round
    df_final_input = df_interest_inputs.round({"Temp_air_2m": 0, "tmrt-ta": 0, "WindSpeed": 1, 'RelativeHumidity': 0})

    return df_final_input


####################################################################
# on doit avoir une base de données contenant les diff param et les deux valeurs de l'UTCI
# Concatenate the input and the output files based on the datetime column and get only the \
# UTCI broede value
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    # Import the data sets
    ## Imprt the data set with the polynomial UTCI
    df_input_UTCI_Polynom = data_management('final_data.xlsx', 'UTCI_TMRT.xlsx')

    ## Import the data set with UTCI of Broede
    df_input_UTCI_Broede = pd.read_csv('final_output.csv', nrows= 10000)

    ## Concatenate the two data sets based on the datetime column
    ### Join
    df_joined_data = pd \
        .merge(left=df_input_UTCI_Polynom, right=df_input_UTCI_Broede, left_on='datetime', right_on='datetime')
    ### Get the columns of interest
    df_filtered_input = df_joined_data[
        ['datetime', 'Temp_air_2m', 'tmrt-ta', 'WindSpeed', 'RelativeHumidity', 'UTCI_input', 'UTCI_broede']]

    ### Select the data based on list of days of reference (we'll not work on all the data)
    ## Convert the strings to datetime in the our pandas dataframe
    df_filtered_input['datetime'] = pd.to_datetime(df_filtered_input['datetime'], errors='coerce')
    ## Add columns for date
    df_filtered_input['date'] = df_filtered_input.datetime.dt.date

    # Selection of days of reference
    ## List of days of reference
    days_of_ref = ['2019-07-25']
    ## Converting the list of days to a date
    days_of_ref_date = []
    for d in days_of_ref:
        days_of_ref_date.append(datetime.datetime.strptime(d, "%Y-%m-%d").date())

    ## Filtering the data based on the list of days of reference
    df_filtered_input_days_of_ref = df_filtered_input[df_filtered_input['date'].isin(days_of_ref_date)]
    # TODO Here we need une boucle pour : pour faire la reg linéaire pour chaque journée à part
    # Get the final input
    df_final_input = df_filtered_input_days_of_ref[
        ['Temp_air_2m', 'tmrt-ta', 'WindSpeed', 'RelativeHumidity', 'UTCI_input']]



    # Multiple Linear Regression
    ####### TODO :  We'll do the same thing twice : first for UTCI_polynom then for UTCI broede \
    # TODO : so we have to change the UTCI_input in the final input with broede UTCI
    ## Define the X and Y related data

    X = df_final_input.iloc[:, :-2].values
    Y = df_final_input.iloc[:, 4].values
    # TODO : here X with the column 4 then we'll have the 5

    ## Splitting the dataset into the Training set and Test set

    from sklearn.model_selection import train_test_split

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Fitting the Multiple Linear Regression in the Training set

    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()
    regressor.fit(X_Train, Y_Train)

    # Predicting the Test set results

    Y_Pred = regressor.predict(X_Test)
    print(Y_Pred)
    
    from sklearn.metrics import r2_score
    score = r2_score(Y_Test, Y_Pred)
    print(score)

