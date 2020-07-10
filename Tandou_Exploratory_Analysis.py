#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Safinez BOUMAIZA
purpose : Describe the UTCI variable and the four different parameters (Ta, tmrt-Ta, Va and RH)
For this study we will be interested in the UTCI value generated by the fast calculation
"""


# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


def data_management(input_file1,input_file2 ):
    '''
    This function aims to gather two dataframes into one based on the datetime column, calculate tmrt-Ta
    :param input_file: name of the data file to import
    :return: One input file (dataframe)
    '''
    # Import Data
    final_data = pd.read_excel(input_file1, indexcol=0, nrows=40000)
    utci_tmrt = pd.read_excel(input_file2, indexcol=0, nrows= 40000)
    utci_tmrt.rename(columns={utci_tmrt.columns[0]: "datetime"}, inplace=True)

    # Jointure
    df_joined_inputs = pd.merge(left=final_data, right=utci_tmrt, left_on='datetime', right_on='datetime')

    # TMRT-Ta calculation
    df_joined_inputs['tmrt-ta'] = df_joined_inputs['T_mrt'] - df_joined_inputs['Temp_air_2m']

    # Creation of the table_offset
    df_interest_inputs = df_joined_inputs[
        ['datetime', 'Temp_air_2m', 'tmrt-ta', 'WindSpeed', 'RelativeHumidity', 'UTCI']]

    ## Arrondir
    df_final_input = df_interest_inputs.round({"Temp_air_2m": 0, "tmrt-ta": 0, "WindSpeed": 1, 'RelativeHumidity': 0})

    ## Trier
    return df_final_input



if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    # Import data
    df_input = data_management('final_data.xlsx', 'UTCI_TMRT.xlsx')

    # Describe Data : General
    print(df_input.describe().round(2))
    print(df_input.isnull())

    # Describe the distribution of each variable
