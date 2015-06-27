# The script MUST contain a function named azureml_main
# which is the entry point for this module.
#
# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
# This module develop the counting features of log events.
# The output is a dataframe with 41 columns:
# Col1: enrollment id
# Cols 2-8: counts of events in Monday to Sunday
# Cols 9-32: counts of events in hour 0-23
# Cols 33-39: counts of event types
# Cols 40-41: counts of source types
from numpy import *
from datetime import datetime
import pandas as pd
def azureml_main(dataframe1 = None, dataframe2 = None):

    # Execution logic goes here
    unique_enrollment = list(set(dataframe1.iloc[:,0]))
# Define dictionaries to map enrollment_id to row indices
    num_unique_enrollment = len(unique_enrollment)
    enrollment_dict = dict(zip(unique_enrollment,range(num_unique_enrollment)))
    numrows = dataframe1.shape[0]
    count_features = zeros((num_unique_enrollment, 41))
    # define dictionaries to map source names and event types to indices
    source_dict = {'browser':0,'server':1}
    event_dict = dict(zip(['access','problem','page_close',\
        'nagivate','video','discussion','wiki'],range(7)))

    for i in range(numrows):
        enrollment_id = dataframe1.iloc[i,0]
        row_index = enrollment_dict[enrollment_id]
        count_features[row_index,0] = enrollment_id
        timestamp_i = float(dataframe1.iloc[i,1])
        dateobj = datetime.fromtimestamp(timestamp_i)
        weekday = dateobj.weekday()
        hour = dateobj.hour
        #weekday is between 0 and 6, where Monday is 0, and Sunday is 6
        count_features[row_index,weekday+1] += 1
# hour is between 0 and 23
        count_features[row_index,hour+8] += 1
        event_index = event_dict[dataframe1.iloc[i,3]]
        source_index = source_dict[dataframe1.iloc[i,2]]
        count_features[row_index,event_index+32] += 1
        count_features[row_index,source_index+39] += 1
    dataframe1 = pd.DataFrame(count_features)

    dataframe1.columns = ['enrollment_id','MonCount',\
        'TueCount','WedCount','ThuCount','FriCount',\
        'SatCount','SunCount','Hr0Count','Hr1Count','Hr2Count',\
        'Hr3Count','Hr4Count','Hr5Count','Hr6Count',\
        'Hr7Count','Hr8Count','Hr9Count','Hr10Count',\
        'Hr11Count','Hr12Count','Hr13Count','Hr14Count',\
        'Hr15Count','Hr16Count','Hr17Count','Hr18Count',\
        'Hr19Count','Hr20Count','Hr21Count','Hr22Count',\
        'Hr23Count','AccCount','ProCount','PagCount',\
        'NagCount','VidCount','DisCount','WikCount',\
        'BroCount','SerCount']
    # If a zip file is connected to the third input port is connected,
    # it is unzipped under ".\Script Bundle". This directory is added
    # to sys.path. Therefore, if your zip file contains a Python file
    # mymodule.py you can import it using:
    # import mymodule

    # Return value must be of a sequence of pandas.DataFrame
    return dataframe1,