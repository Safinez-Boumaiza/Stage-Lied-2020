#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import packages
import pandas as pd
import numpy as np


def data_management():
    '''
    This function aims to gather two dataframes into one based on the datetime column, calculate tmrt-Ta
    :return: Table
    '''
    # Import Data
    final_data = pd.read_excel('final_data.xlsx', indexcol=0)
    utci_tmrt = pd.read_excel('UTCI_TMRT.xlsx', indexcol=0)
    utci_tmrt.rename(columns={utci_tmrt.columns[0]: "datetime"}, inplace=True)

    # Jointure
    df_joined_inputs = pd.merge(left=final_data, right=utci_tmrt, left_on='datetime', right_on='datetime')
    # data = df_joined_inputs.drop(df_joined_inputs.columns[[9]], axis = 'columns')

    # TMRT-Ta calculation
    df_joined_inputs['tmrt-ta'] = df_joined_inputs['T_mrt'] - df_joined_inputs['Temp_air_2m']

    # Creation of the table_offset
    df_interest_inputs = df_joined_inputs[
        ['datetime', 'Temp_air_2m', 'tmrt-ta', 'WindSpeed', 'RelativeHumidity', 'UTCI']]

    ## Arrondir TODO  vaut mieux le mettre dans la fonction
    df_final_input = df_interest_inputs.round({"Temp_air_2m": 0, "tmrt-ta": 0, "WindSpeed": 1, 'RelativeHumidity': 0})

    ## Trier
    # table_offset_final = table_offset_round.sort_values(by = ['Temp_air_2m', 'tmrt-ta','WindSpeed'])
    return df_final_input


"""
# Match UTCI function
def match_utci (df_input, df_lookup_table):
    '''

    :param df_input: Data frame of experiment data
    :param df_lookup_table: Look up table
    :return: data frame containing Datetime, UTCI_exp and UTCI_ref values
    '''
    merged_data = pd.merge(
        df_input,
        df_lookup_table,
        left_on=['Temp_air_2m','tmrt-ta', 'WindSpeed', 'RelativeHumidity'],
        right_on=['Ta','Tr-Ta','va', 'rH']
    )

    merged_data['UTCI_ref'] = merged_data['Offset'] + merged_data['Temp_air_2m']
    result = merged_data[['datetime','UTCI', 'UTCI_ref']]
    # Ou bien on rajoute les données d'entrée
    return result
"""


def get_closest_value(x, df, col):
    '''
    Get the equal or closest value to x in the column col of the dataframe df
    :param x: X is the value we are looking to find or to find it's closest value
    :param df: A dataframe
    :param col: A column of the dataframe df
    :return:The equal or closest value to x
    '''
    # get the biggest value in the column that is smaller than x
    # get the smallest value in the column that is bigger than x
    df2 = df[col].to_list()

    near_up = 0
    near_low = 0

    list_up = [float(i) for i in df2 if float(i) >= float(x)]
    if (len(list_up) >= 1):
        near_up = min(list_up)
    else:
        return max(df2)
    list_down = [float(i) for i in df2 if float(i) < float(x)]
    if (len(list_down) >= 1):
        near_low = max(list_down)
    else:
        return min(df2)

    if abs(near_up - x) < abs(near_low - x):
        return near_up
    else:
        return near_low


def get_utci(table, values):
    '''
    get_utci returns the
    :param table: lookup tab
    :param values: list of values
    :return:
    '''
    # Segment the look up table into 4 small parts based on each input value
    df1 = table.loc[(table['Ta'] == values[0])]
    df2 = df1.loc[(table['Tr-Ta'] == values[1])]
    df3 = df2.loc[(table['va'] == values[2])]
    df_final = df3.loc[(table['rH'] == values[3])]
    return df_final.iat[0, 4]


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    df_input = data_management()
    df_input = df_input.head(100)
    df_lookup_table = pd \
        .read_csv('ESM_4_Table_Offset.Dat', skiprows=23, sep='\t') \
        .drop('pa', 1)
    # test Calculate UTCI value
    # df_lookup_table["Offset"] = df_lookup_table["Offset"] +df_lookup_table["Ta"]
    output = pd.DataFrame(columns=('datetime', 'offset'))
    # loop through input data
    for x in range(len(df_input.index)):
        df_lookup_temp = df_lookup_table
        # will contain the four closest values found in lookup table
        closest_values = []
        # loop through the four columns to match for this particular row
        for index, col in enumerate(['Ta', 'Tr-Ta', 'va', 'rH']):
            # add the closest value for this column for this row to the list
            closest_values.append(get_closest_value(df_input.iat[x, index + 1], df_lookup_temp, col))
            df_lookup_temp = df_lookup_temp[(df_lookup_temp[col] == closest_values[index])]

        # get utci value using the four values in list
        utci = get_utci(df_lookup_table, closest_values)
        output.loc[x] = [df_input.iat[x, 0], utci]
    print(output.head(5))

# TODO  Plot comparaison UTCI-UTCIref for
# indices = [i for i, x in enumerate(df2) if x == near_low]