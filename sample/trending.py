# The script MUST contain a function named azureml_main
# which is the entry point for this module.
#
# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame

# This module extracts 30 features from the log event data, including weekly trending slope, number of sessions,
# duration of sessions, average number of events in each session, etc. More details, please read the comments before
# function extract_enrollment_log()
from numpy import *
import datetime, time
import pandas as pd

#This function extracts statistics for each session in an enrollment, such as average session duration, standard deviation
# of session duration, and min and max session duration. All in unit hours.
# Inputs: data is an array with 5 columns,
# Col1: year,
# Col2: week number,
# Col3: session id when new session starts if two events are apart for more than 3 hours,
# Col4: session id when new session starts if two events are apart for more than 1 hour,
# Col5: time stamp of the event
# Note: session id starts from 1
# The second input is the column index of the session id. When calculating the statistics of 3-hour defined sessions,
# session id index= 2, when calculating the statistics of 1-hour defined sessions, session id index=3,
# Outputs: session_stat, a list of 4 float values, in the unit of hours.
# Value1: average duration of sessions
# Value2: if session number>1, standard deviation of session duration. Otherwise, 0
# Value3: maximal session duration
# Value4: minimal session duration
def extract_session_stat(data,sessionid_index):
    numrows = data.shape[0]
    sessionindex = 0
    numsessions = max(data[:,sessionid_index])
    session_min = [float(inf)]*numsessions
    session_max = [0]*numsessions
    session_stat = [0]*4
    for i in range(numrows):
        sessionindex = int(data[i,sessionid_index])-1
        if session_min[sessionindex] > data[i,4]:
            session_min[sessionindex] = data[i,4]
        if session_max[sessionindex] < data[i,4]:
            session_max[sessionindex] = data[i,4]
    session_duration = (array(session_max)-array(session_min))/3600

    session_stat[0] = mean(session_duration)
    if numsessions > 1:
        session_stat[1] = std(session_duration)
    session_stat[2] = max(session_duration)
    session_stat[3] = min(session_duration)
    return session_stat

# This function extracts statistics for each enrollment, and call function extract_session_stat to extract statistics for sessions
# Input:  an array with two columns:
# Col1: enrollment id
# Col2: timestamps of log events, in ascending order
# Output:  A list of 31 log event statistics, These are the output features of this Python module
# Value1: enrollment_id
# Value2: trending slope of the weekly number of events within the enrollment
# Value3: number of events in the last week of the enrollment
# Value4: number of events in the first week of the enrollment
# Value5: number of events in the week before the last week of the enrollment
# Value6: average weekly number of events in the enrollment period
# Value7: standard deviation of the weekly number of events in the enrollment period
# Value8: maximal weekly number of events in the enrollment period
# Value9: minimal weekly number of events in the enrollment period
# Value10: month (1-12) of the first event in the enrollment
# Value11: month of the last event in the enrollment
# Value12: number of 3-hour defined sessions in the enrollment
# Value13: average number of events in 3-hour defined sessions in the enrollment
# Value14: standard deviation of the number of events in 3-hour defined sessions in the enrollment
# Value15: maximal number of events in 3-hour defined sessions in the enrollment
# Value16: minimal number of events in 3-hour defined sessions in the enrollment
# Values 17 & 18: coefficients b and c in the polynomial model y = a + bx + cx**2, where x is week number (all start from 0), and
# y is the weekly number of events
# Value19: number of 1-hour defined sessions in the enrollment
# Value20: average number of events in 1-hour defined sessions in the enrollment
# Value21: standard deviation of the number of events in 1-hour defined sessions in the enrollment
# Value22: maximal number of events in 1-hour defined sessions in the enrollment
# Value23: minimal number of events in 1-hour defined sessions in the enrollment
# Values 24-27: statistics of 3-hour defined sessions
# Values28-31: statistics of 1-hour defined sessions
def extract_enrollment_log(log):
    num_events = len(log)
    chunk_data = zeros((num_events,5))
    features = [0]*31
    features[0] = log[0][0]
    previous_stamp_3hr = 0
    previous_stamp_1hr = 0
    session_id_3hr = 0
    session_id_1hr = 0
    min_year = 3000
    max_year = 1000
    max_week = 0
# min_timestamp and max_timestamp will be used to calculate the duration of
# the enrollment
    min_timestamp = float('inf')
    max_timestamp = 0
    for i in range(num_events):
        timestamp_i = log[i][1]
        if min_timestamp > timestamp_i:
            min_timestamp = timestamp_i
        if max_timestamp < timestamp_i:
            max_timestamp = timestamp_i
# Assign session id to a record. If two consecutive records are apart for over 3 hours for longer session definition
# (or 1 hour for short session definition), a new session id starts
        if timestamp_i - previous_stamp_3hr >= 10800:
            session_id_3hr += 1
        if timestamp_i - previous_stamp_1hr >= 3600:
            session_id_1hr += 1
        previous_stamp_3hr = timestamp_i
        previous_stamp_1hr = timestamp_i
        chunk_data[i,4] = timestamp_i
        chunk_data[i,2] = session_id_3hr
        chunk_data[i,3] = session_id_1hr
        date_object = datetime.datetime.fromtimestamp(timestamp_i)
        # The year and week number of each record
# These two variables will be used to calculate the trending of events in
# each week.
        year_num = date_object.year
        week_num = int(date_object.strftime("%U"))
        if year_num > max_year:
            max_year = year_num
        if year_num < min_year:
            min_year = year_num
        if week_num > max_week:
            max_week = week_num
        chunk_data[i,0] = year_num
        chunk_data[i,1] = week_num
# if the enrollment is in two years (crossing the new year's day)
# we need to reassign the week number in the second year
# to make sure that the week number is ascending with dates
    if max_year > min_year:
        week_last_day = datetime.date(min_year, 12, 31).isocalendar()[1]
        if week_last_day == 1:
            week_last_day = datetime.date(min_year, 12, 24).isocalendar()[1]
        max_year_index = [i for i, j in enumerate(chunk_data[:,0]) if j == max_year]
        chunk_data[max_year_index,1] += week_last_day
        min_year_index = [i for i, j in enumerate(chunk_data[:,0]) if j == min_year]
        week1_index = [i for i, j in enumerate(chunk_data[:,1]) if j == 1]
        intersection = list(set(min_year_index).intersection(week1_index))
        if len(intersection)>0:
            chunk_data[intersection,1] += week_last_day
        max_week = max(chunk_data[:,1])
    min_week = min(chunk_data[:,1])
    num_weeks = int(max_week - min_week)
    events_count = [0]*(num_weeks+1)
    num_sessions_3hr = max(chunk_data[:,2])
    num_sessions_1hr = max(chunk_data[:,3])
    session_count_3hr = [0]*num_sessions_3hr
    session_count_1hr = [0]*num_sessions_1hr
    for i in range(num_events):
        events_count[int(chunk_data[i,1]-min_week)] += 1
        session_count_3hr[int(chunk_data[i,2]-1)] += 1
        session_count_1hr[int(chunk_data[i,3]-1)] += 1
    if num_weeks > 0:
        weeks = range(num_weeks+1)
        events_count = array(events_count)
        A = vstack([weeks, ones(len(weeks))]).T
        m, c = linalg.lstsq(A, events_count)[0]
        features[1] = m
    if num_weeks > 1:
        weeks = range(num_weeks+1)
        events_count = array(events_count)
        #fit a polynomial model y = a + bx + cx**2
        z = polyfit(weeks, events_count, 2)
        features[16] = z[1]
        features[17] = z[2]
    features[2] = events_count[-1]
    features[3] = events_count[0]
    if num_weeks > 0:
        features[4] = events_count[-2]
    features[5] = mean(events_count)
    if num_weeks > 0:
        features[6] = std(events_count)
    features[7] = max(events_count)
    features[8] = min(events_count)
    date_object = datetime.datetime.fromtimestamp(min_timestamp)
    features[9] = date_object.month
    date_object = datetime.datetime.fromtimestamp(max_timestamp)
    features[10] = date_object.month
    features[11] = num_sessions_3hr
    features[12] = mean(session_count_3hr)
    if num_sessions_3hr > 1:
        features[13] = std(session_count_3hr)
    features[14] = max(session_count_3hr)
    features[15] = min(session_count_3hr)
    features[18] = num_sessions_1hr
    features[19] = mean(session_count_1hr)
    if num_sessions_1hr > 1:
        features[20] = std(session_count_1hr)
    features[21] = max(session_count_1hr)
    features[22] = min(session_count_1hr)
    session_stat_3hr = extract_session_stat(chunk_data,2)
    session_stat_1hr = extract_session_stat(chunk_data,3)
    features[23:27] = session_stat_3hr
    features[27:31] = session_stat_1hr
    return features

# azureml_main function is the main function that is called during execution.
def azureml_main(dataframe1 = None, dataframe2 = None):

    num_obs = dataframe1.shape[0]
    num_enrollment = len(set(dataframe1['enrollment_id']))
    print("There are %d unique enrollment ids."%num_enrollment)
    # The output will be a data frame which has num_enrollment rows, and 31 columns, where the first column is the enrollment id,
    # and the remaining 30 columns are features
    output_df = zeros((num_enrollment,31))
    enrollment_log = []
    previous_id = -1
    enrollment_index = 0
    #Partition log records by enrollment_id, and then extract statistics for each
#enrollment
    for i in range(num_obs):
      current_id = dataframe1.iloc[i][0]
      # If a new enrollment_id starts, indicating that we have collected all log records
# for that enrollment_id, then we can start processing log records of the enrollment_id
      if current_id != previous_id:
          if i > 0:
              output_df[enrollment_index,:] = extract_enrollment_log(enrollment_log)
              enrollment_index += 1
          enrollment_log = []
          previous_id = current_id
      enrollment_log.append(dataframe1.iloc[i])
    output_df[enrollment_index,:] = extract_enrollment_log(enrollment_log)
# The output of the Execute Python Script has to be a Pandas data frame. Column names are added to the data frame
# for the convenience of referring to columns in the consequential modules.
    dataframe1 =  pd.DataFrame(output_df)
    dataframe1.columns = ['enrollment_id','event_trend','events_last_week',\
                          'events_first_week','events_second_last_week',\
                          'weekly_avg','weekly_std','max_weekly_count','min_weekly_count',\
                          'first_event_month','last_event_month','session_count_3hr',\
                          'session_avg_3hr','session_std_3hr','session_max_3hr','session_min_3hr',\
                          'quadratic_b','quadratic_c','session_count_1hr',\
                          'session_avg_1hr','sessioin_std_1hr','sessioin_max_1hr',\
                          'session_min_1hr','session_dur_avg_3hr','session_dur_std_3hr',\
                          'sessioin_dur_max_3hr','session_dur_min_3hr','sessioin_dur_avg_1hr',\
                          'session_dur_std_1hr','session_dur_max_1hr','session_dur_min_1hr']
    return dataframe1
    # If a zip file is connected to the third input port is connected,
    # it is unzipped under ".\Script Bundle". This directory is added
    # to sys.path. Therefore, if your zip file contains a Python file
    # mymodule.py you can import it using:
    # import mymodule

    # Return value must be of a sequence of pandas.DataFrame
