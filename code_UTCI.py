#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Safinez Boumaiza
"""

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
# import matplotlib.dates as mdates
# import matplotlib.cbook as cbook


def data_management(input_file1,input_file2 ):
    '''
    This function aims to gather two dataframes into one based on the datetime column, calculate tmrt-Ta
    :param input_file: name of the data file to import
    :return: One input file (dataframe)
    '''
    # Import Data
    final_data = pd.read_excel(input_file1, indexcol=0)
    utci_tmrt = pd.read_excel(input_file2, indexcol=0)
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


def get_closest_value(x, df_table, col):
    '''
    Get the equal or closest value to x in the column col of the dataframe df
    :param x: X is the value we are looking to find or to find it's closest value
    :param df: The lookup table
    :param col: A column of the dataframe df
    :return:The equal or closest value to x
    '''
    # get the biggest value in the column that is smaller than x
    # get the smallest value in the column that is bigger than x
    df_list = df_table[col].to_list()

    near_up = 0
    near_low = 0

    list_up = [float(i) for i in df_list if float(i) >= float(x)]
    if (len(list_up) >= 1):
        near_up = min(list_up)
    else:
        return max(df_list)
    list_down = [float(i) for i in df_list if float(i) < float(x)]
    if (len(list_down) >= 1):
        near_low = max(list_down)
    else:
        return min(df_list)

    if abs(near_up - x) < abs(near_low - x):
        return near_up
    else:
        return near_low


def get_utci(df_look_up_table, values):
    '''
    get_utci returns the
    :param table: lookup tab
    :param values: list of input values
    :return: The UTCI Value corresponding to all the input values
    '''

    # Segment the look up table into 4 small parts based on each input value

    df1 = df_look_up_table.loc[(df_look_up_table['Ta'] == values[0])]
    df2 = df1.loc[(df_look_up_table['Tr-Ta'] == values[1])]
    df3 = df2.loc[(df_look_up_table['va'] == values[2])]
    df_final = df3.loc[(df_look_up_table['rH'] == values[3])]

    return df_final.iat[0, 4]


def plot_utci(df_to_plot, list_d_date):
    '''

    :param df_to_plot: Data frame containing UTCI values (two different methods)/
                       date, hour and datetime columns
    :param list_d_date: The list of days of reference we need to focus on
    :return: The function returns for each day of reference a plot of the two/
    UTCI values and a plot for the difference between them
    '''

    # Filter our data based on the list of days of reference
    df_to_plot_filtered = df_to_plot[df_to_plot['date'].isin(list_d_date)]

    # Create a plot per day from the list of days
    # Scroll through all the days
    for d in range(len(list_d_date)):
        df_to_plot_filtered_day = df_to_plot_filtered.loc[df_to_plot_filtered['date'] == list_d_date[d]]
        df_to_plot_filtered_day = df_to_plot_filtered_day.groupby(['hour'], as_index=False).mean()

        # Plots
        # Plot 1
        x = df_to_plot_filtered_day['hour']
        y1 = df_to_plot_filtered_day['UTCI_input']
        y2 = df_to_plot_filtered_day['UTCI_broede']
        y3 = df_to_plot_filtered_day['Diff_UTCI']
        plt.plot(x, y1, label='Polynomial')
        plt.plot(x, y2, label='Broede')
        plt.plot(x, y3, label='Difference')
        plt.legend()
        plt.title("Comparaison between two methods of calculation of the UTCI value")
        plt.xlabel("Day of reference")
        plt.ylabel("UTCI values")

        plt.show()

        # This part is in case we want to plot the difference only
    """
        # Plot 2
        xx = df_to_plot_filtered_day['hour']
        yy1 = df_to_plot_filtered_day['Diff_UTCI']
        plt.plot(xx, yy1, label='Difference')
        plt.legend()
        plt.title("Comparaison between two methods of calculation of the UTCI value")
        plt.xlabel("Day of reference")
        plt.ylabel("Difference between the two UTCI values")

        plt.show()
    """

def mean_day_plot (df_to_manage, list_d_date):
    '''
    This function allows us to create a mean day from a list of days
    and to plot the result
    :param df_to_manage: The dataframe containing the datetime data
    :param list_d_date: A list of days converted to date type
    :return: A plot for the mean day of all the days in the the list_d_date
    '''
    # Create and plot a mean day for the different UTCI values
    df_for_mean_day = df_to_manage[df_to_manage['date'].isin(list_d_date)]
    df_mean_day = df_for_mean_day.groupby(['hour', 'minute'], as_index=False).mean()
    # Create a new column for the full time
    df_mean_day['full_time'] = df_mean_day['hour'].astype(str)+':'+df_mean_day['minute'].astype(str)


    x = df_mean_day['full_time']
    y1 = df_mean_day['UTCI_broede']
    y2 = df_mean_day['UTCI_input']
    plt.plot(x, y1, "-", color = 'red' ,label='Broede')
    plt.plot(x, y2, "-", color = 'green', label='Polynomia')
    # plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.ylim(0, 50)
    plt.legend()
    plt.title("Mean day")
    plt.xlabel("Hour")
    plt.ylabel("UTCI value")

    plt.show()

'''
    ######### test ########

    idx = pd.timedelta_range('0', '23', freq = '1H')
    df = pd.DataFrame(np.cumsum(np.random.randn(len(idx), 2),0),
                  index = idx)



    fig, ax = plt.subplots()
    ax.plot(df.index, x, color='black')
    ax2 = ax.twinx()
    ax2.plot(df.index, y1, color='indigo')

    # years = mdates.YearLocator()  # every year
    # months = mdates.MonthLocator()  # every month
    # years_fmt = mdates.DateFormatter('%Y')

    hours = mdates.HourLocator(interval=1) # Every hour
    h_fmt = mdates.DateFormatter('%H:%M:%S')
    # format the ticks
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(h_fmt)

    # round to nearest years. => round the nearest hour
    # hour_min = np.datetime64(df_mean_day['hour'][0], 'Y')
    # hour_max = np.datetime64(df_mean_day['hour'][-1], 'Y') + np.timedelta64(1, 'Y')
    # ax.set_xlim(hour_min, hour_max)

    # format the coords message box
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
    ax.grid(True)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    ######### test #########
'''

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    # Import data
    ## Import the input dataframe
    df_input = data_management('final_data.xlsx', 'UTCI_TMRT.xlsx')


    ## Import the Look up table
    df_lookup_table = pd \
        .read_csv('ESM_4_Table_Offset.Dat', skiprows=23, sep='\t') \
        .drop('pa', 1)

    # Calculate UTCI of broede value
    df_lookup_table["Offset"] = df_lookup_table["Offset"] + df_lookup_table["Ta"]
    df_lookup_table.rename(columns={'Offset': 'UTCI_broede'}, inplace=True)
    # Create an output dataframe with the two UTCI values and the datetime value
    output = pd.DataFrame(columns=('datetime', 'UTCI_input', 'UTCI_broede'))

    # Look for the UTCI equivalent value to our experimental data
    ## loop through input data
    for x in range(len(df_input.index)):
        df_lookup_temp = df_lookup_table

        ## Will contain the four closest values found in lookup table
        closest_values = []

        ## Loop through the four columns to match for this particular row
        for index, col in enumerate(['Ta', 'Tr-Ta', 'va', 'rH']):
            # add the closest value for this column for this row to the list
            closest_values.append(get_closest_value(df_input.iat[x, index + 1], df_lookup_temp, col))
            df_lookup_temp = df_lookup_temp[(df_lookup_temp[col] == closest_values[index])]

        ## Get utci value using the four values in list
        utci = get_utci(df_lookup_table, closest_values)
        output.loc[x] = [df_input.iat[x, 0], df_input.iat[x, 5], utci.round(1)]
    ## Save the results as a csv file
    """
     output.to_csv('final_output.csv', index=False)
    """

    # Output data management
    # Convert the strings to datetime in the our pandas dataframe
    output['datetime'] = pd.to_datetime(output['datetime'], errors='coerce')

    # Add a column for the difference between the two UTCI values
    output["Diff_UTCI"] = output["UTCI_input"] - output["UTCI_broede"]

    # Add three columns, one for the date, one for the hour and a column for the time
    output['date'] = output.datetime.dt.date
    output['hour'] = output.datetime.dt.hour
    output['time'] = output.datetime.dt.time
    output['minute'] = output.datetime.dt.minute
    # print(output.head(5))
    # print(output.describe())

    # Plot the data based on a list of days of reference
    days_of_ref = ['2019-08-24', '2019-08-30']


    # Converting the list of days to a date
    days_of_ref_date = []
    for d in days_of_ref:
        days_of_ref_date.append(datetime.datetime.strptime(d, "%Y-%m-%d").date())
    # Create the plots
    #plot_utci(output, days_of_ref_date)
    mean_day_plot(output,days_of_ref_date)
