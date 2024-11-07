# The function provided below is used for inferencing the model
import os
import pandas as pd
import re
from dotenv import load_dotenv
import numpy as np
import joblib

load_dotenv()

EYETRACKING_DATA_PATH = os.getenv('EYETRACKING_DATA_PATH')
HEARTRATE_DATA_PATH = os.getenv('HEARTRATE_DATA_PATH')
MODEL_PATH = os.getenv('MODEL_PATH')

def predict_vo2max(EYETRACKING_DATA_PATH, HEARTRATE_DATA_PATH, MODEL_PATH):
    """
    Load Eyetracking and Heartrate CSV data and predict the VO2MAX from both data

    Parameters:
    ----------
    EYETRACKING_DATA_PATH : string
        Directory Path of Eyetracking CSV file
    HEARTRATE_DATA_PATH : string
        Path of Heartrate CSV file
    MODEL_PATH : string
        Path of Model .pkl file

    Returns:
    -------
    predictions : float 
        The value of VO2MAX

    """
    # Set the directory path where the CSV files are located
    directory_path = EYETRACKING_DATA_PATH

    # Create an empty list to store dataframes
    dataframes = []

    # Define a regular expression to extract name, date, category, and random number from filenames
    filename_pattern = re.compile(r'(?P<name>[\w\-]+)-(?P<date>\d{4}-\d{2}-\d{2})_(?P<category>\w+)_(?P<random_number>\d+)\.csv')

    # Iterate through each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            # Match the pattern to extract metadata from the filename
            match = filename_pattern.match(filename)
            if match:
                # Extract the components from the filename
                name = match.group('name')
                date = match.group('date')
                category = match.group('category')
                random_number = match.group('random_number')
                
                # Load the CSV file into a dataframe
                file_path = os.path.join(directory_path, filename)
                df = pd.read_csv(file_path)
                
                # Add the extracted components as constant columns
                df['name'] = name
                df['date'] = date
                df['category'] = category
                df['id'] = filename.replace('.csv', '')
                df['path_name'] = filename.replace('.csv', '')
                
                # Append the dataframe to the list
                dataframes.append(df)

    # Concatenate all dataframes into one
    df_eyetracking = pd.concat(dataframes, ignore_index=True)

    df_eyetracking_sp = df_eyetracking[df_eyetracking['category']=='SP']

    df_eyetracking_sem = df_eyetracking[df_eyetracking['category']=='SEM']


    # Load Heartrate
    df_heartrate = pd.read_csv(HEARTRATE_DATA_PATH)

    # Merge Dataframe
    df_eyetracking_sp = pd.merge(df_eyetracking_sp,df_heartrate, left_on='path_name', right_on='SP_result', how='left')

    df_eyetracking_sem = pd.merge(df_eyetracking_sem,df_heartrate, left_on='path_name', right_on='SEM_result', how='left')

    df_eyetracking_sp = df_eyetracking_sp.drop(['path_name','SP_result','SEM_result'], axis =1)
    df_eyetracking_sem = df_eyetracking_sem.drop(['path_name','SP_result','SEM_result'], axis =1)


    # Preprocessing
    df_sp = df_eyetracking_sp
    df_sem = df_eyetracking_sem

    # Drop Null Value
    df_sp = df_sp.dropna()
    df_sem = df_sem.dropna()

    # Feature Engineering
    # Counting Distance
    # Smooth Pursuit
    df_sp['sp_distance'] = (df_sp['MovingTarget_X'] - df_sp['EyeTracker_X'])**2 + (df_sp['MovingTarget_Y'] - df_sp['EyeTracker_Y'])**2

    df_sp['sp_distance'] = np.sqrt(df_sp['sp_distance'])


    screen_width = 1920
    screen_height = 1080

    screen_center = [screen_width//2, screen_height//2] 

    df_sp_angle = df_sp.copy()

    df_sp_angle['vector_e_x'] = df_sp_angle['EyeTracker_X'] - screen_center[0]
    df_sp_angle['vector_e_y'] = df_sp_angle['EyeTracker_Y'] - screen_center[1]

    df_sp_angle['vector_g_x'] = df_sp_angle['MovingTarget_X'] - screen_center[0]
    df_sp_angle['vector_g_y'] = df_sp_angle['MovingTarget_Y'] - screen_center[1]

    # df_sp['vector_e'] = [df_sp['vector_e_x'], df_sp['vector_e_y']]
    # df_sp['vector_g'] = [df_sp['vector_g_x'], df_sp['vector_g_y']]

    df_sp_angle['vector_e_length'] = df_sp_angle['vector_e_x']**2 + df_sp_angle['vector_e_y']**2
    df_sp_angle['vector_e_length'] = np.sqrt(df_sp_angle['vector_e_length'])

    df_sp_angle['vector_g_length'] = df_sp_angle['vector_g_x']**2 + df_sp_angle['vector_g_y']**2
    df_sp_angle['vector_g_length'] = np.sqrt(df_sp_angle['vector_g_length'])

    df_sp_angle['vector_e_dot_g'] = df_sp_angle['vector_e_x'] * df_sp_angle['vector_g_x'] + df_sp_angle['vector_e_y'] * df_sp_angle['vector_g_y']

    df_sp_angle['sp_angle'] = np.arccos(df_sp_angle['vector_e_dot_g'] / (df_sp_angle['vector_e_length'] * df_sp_angle['vector_g_length']))

    df_sp['sp_angle'] = df_sp_angle['sp_angle']

    df_sp_row_id = df_sp[['id','row_id']].drop_duplicates()
    df_sp = df_sp[['id', 'HR_before', 'HR_after', 'sp_distance', 'sp_angle']]

    # Group by 'category' and apply aggregation functions
    df_sp_sum = df_sp.groupby('id').agg({
        'sp_distance': 'sum',    # Sum the values in 'values_1'
        'sp_angle': 'sum'    # Calculate the mean of 'values_2'
    })

    # Group by 'category' and apply aggregation functions
    df_sp_average = df_sp.groupby('id').agg({
        'sp_distance': 'mean',    # Sum the values in 'values_1'
        'sp_angle': 'mean'    # Calculate the mean of 'values_2'
    })

    # Group by 'category' and apply aggregation functions
    df_sp_median = df_sp.groupby('id').agg({
        'sp_distance': 'median',    # Sum the values in 'values_1'
        'sp_angle': 'median'    # Calculate the mean of 'values_2'
    })

    df_sp = df_sp[['id', 'HR_before', 'HR_after']]
    df_sp = df_sp.groupby('id').agg('mean')

    df_sp['sp_total_distance'] = df_sp_sum['sp_distance']
    df_sp['sp_average_distance'] = df_sp_average['sp_distance']
    df_sp['sp_median_distance'] = df_sp_median['sp_distance']

    df_sp['sp_total_angle'] = df_sp_sum['sp_angle']
    df_sp['sp_average_angle'] = df_sp_average['sp_angle']
    df_sp['sp_median_angle'] = df_sp_median['sp_angle']

    df_sp = df_sp.reset_index()
    df_sp = pd.merge(df_sp, df_sp_row_id, on='id', how='inner')

    # Saccadic Eye Movement
    # Convert the Unix timestamp to a normal timestamp (assuming it's in microseconds)
    df_sem['Timestamp'] = df_sem['Timestamp'] //100  # Convert from microseconds to seconds
    df_sem['datetime_timestamp'] = pd.to_datetime(df_sem['Timestamp'], unit='ms', utc=True)  # Convert to datetime

    # Filter the moving target to keep only rows with max or min values 
    max_value = df_sem['MovingTarget_X'].max()
    min_value = df_sem['MovingTarget_X'].min()

    df_sem = df_sem[(df_sem['MovingTarget_X'] == max_value) | (df_sem['MovingTarget_X'] == min_value)]

    # Create a discrete signal
    df_sem['MovingTarget_X_discrete'] = np.where(df_sem['MovingTarget_X'] == max_value, max_value, 
                                np.where(df_sem['MovingTarget_X'] == min_value, min_value, np.nan))

    df_sem['MovingTarget_X_discrete'] = df_sem['MovingTarget_X_discrete'].replace(max_value, 'right')
    df_sem['MovingTarget_X_discrete'] = df_sem['MovingTarget_X_discrete'].replace(min_value, 'left')

    def check_if_eye_is_inside_the_circle(MovingTarget_X, MovingTarget_Y, EyeTracker_X, EyeTracker_Y, circle_radius):
        position = (EyeTracker_X - MovingTarget_X)**2 + (EyeTracker_Y - MovingTarget_Y)**2
        if position <= circle_radius**2:
            return True
        else:
            return False
        
    df_sem_row_id = df_sem[['id','row_id']].drop_duplicates()
    all_latency = []
    discrete = ''
    counting = False
    start_time_change = 0
    circle_radius = 20

    for id in df_sem.id.unique():
        latency = []
        df_sem_id = df_sem[df_sem['id'] == id ]
        for row in df_sem_id.itertuples():
            # Check if the previous discrete is changing
            if discrete != row.MovingTarget_X_discrete:
                discrete = row.MovingTarget_X_discrete
                counting = True
                start_time_change = row.datetime_timestamp

            # If counting and eye is inside the circle, calculate latency
            if counting and check_if_eye_is_inside_the_circle(row.MovingTarget_X, row.MovingTarget_Y, row.EyeTracker_X, row.EyeTracker_Y, circle_radius):
                latency_delta = row.datetime_timestamp - start_time_change  # This is a timedelta object
                latency_in_milliseconds = latency_delta.total_seconds() * 1000  # Convert timedelta to milliseconds
                latency.append(latency_in_milliseconds)  # Append latency in milliseconds
                counting = False  # Stop counting after latency is recorded
                start_time_change = 0  # Reset start time

        all_latency.append(latency)

    sum_latency_list = []
    mean_latency_list = []
    median_latency_list = []

    for latency in all_latency:
        sum_latency_list.append(np.sum(latency))
        mean_latency_list.append(np.mean(latency))
        median_latency_list.append(np.median(latency))

    df_latency = pd.DataFrame()
    df_latency['id'] = df_sem.id.unique()
    df_latency['total_latency'] = sum_latency_list
    df_latency['mean_latency'] = mean_latency_list
    df_latency['median_latency'] = median_latency_list

    df_latency = pd.merge(df_latency, df_sem_row_id, on='id', how='left')

    df_sp = df_sp.reset_index()
    df_engineered = pd.merge(df_sp, df_latency, left_on='row_id', right_on='row_id',how='inner')

    df_engineered = df_engineered.drop(['index', 'id_x', 'id_y'], axis = 1)

    df_engineered['vo2_max'] = 15.3 * df_engineered['HR_after'] / df_engineered['HR_before']

    # Feature Selection
    df_engineered = df_engineered.dropna()

    df_engineered = df_engineered[['HR_before','sp_median_distance','sp_median_angle','median_latency']]

    # Inference
    model = joblib.load(MODEL_PATH)

    prediction = model.predict(df_engineered.iloc[0:1])

    print("Prediction:", prediction[0])

    return prediction[0]

def predict_vo2max_v2(FOLDER_PATH, id, model_path, HR_before, age, weight, height, sex):
    """
    Load Eyetracking and Heartrate CSV data and predict the VO2MAX from both data

    Parameters:
    ----------
    FOLDER_PATH : string
        Directory of Path contains folder with id
    id : string
        the id
    model_path : string
        Path of Model .pkl file

    Returns:
    -------
    predictions : float 
        The value of VO2MAX

    """
    # Set the directory path where the CSV files are located
    directory_path = FOLDER_PATH + '/' + id

    df_eyetracking_sem = pd.read_csv(directory_path + '/SEM.csv')
    df_eyetracking_sp = pd.read_csv(directory_path + '/SP.csv')

    # Preprocessing
    df_sp = df_eyetracking_sp
    df_sem = df_eyetracking_sem

    # Drop Null Value
    df_sp = df_sp.dropna()
    df_sem = df_sem.dropna()

    df_sp['row_id'] = 1
    df_sem['row_id'] = 1

    # Feature Engineering
    # Counting Distance
    # Smooth Pursuit
    df_sp['sp_distance'] = (df_sp['MovingTarget_X'] - df_sp['EyeTracker_X'])**2 + (df_sp['MovingTarget_Y'] - df_sp['EyeTracker_Y'])**2

    df_sp['sp_distance'] = np.sqrt(df_sp['sp_distance'])


    screen_width = 1920
    screen_height = 1080

    screen_center = [screen_width//2, screen_height//2] 

    df_sp_angle = df_sp.copy()

    df_sp_angle['vector_e_x'] = df_sp_angle['EyeTracker_X'] - screen_center[0]
    df_sp_angle['vector_e_y'] = df_sp_angle['EyeTracker_Y'] - screen_center[1]

    df_sp_angle['vector_g_x'] = df_sp_angle['MovingTarget_X'] - screen_center[0]
    df_sp_angle['vector_g_y'] = df_sp_angle['MovingTarget_Y'] - screen_center[1]

    # df_sp['vector_e'] = [df_sp['vector_e_x'], df_sp['vector_e_y']]
    # df_sp['vector_g'] = [df_sp['vector_g_x'], df_sp['vector_g_y']]

    df_sp_angle['vector_e_length'] = df_sp_angle['vector_e_x']**2 + df_sp_angle['vector_e_y']**2
    df_sp_angle['vector_e_length'] = np.sqrt(df_sp_angle['vector_e_length'])

    df_sp_angle['vector_g_length'] = df_sp_angle['vector_g_x']**2 + df_sp_angle['vector_g_y']**2
    df_sp_angle['vector_g_length'] = np.sqrt(df_sp_angle['vector_g_length'])

    df_sp_angle['vector_e_dot_g'] = df_sp_angle['vector_e_x'] * df_sp_angle['vector_g_x'] + df_sp_angle['vector_e_y'] * df_sp_angle['vector_g_y']

    df_sp_angle['sp_angle'] = np.arccos(df_sp_angle['vector_e_dot_g'] / (df_sp_angle['vector_e_length'] * df_sp_angle['vector_g_length']))

    df_sp['sp_angle'] = df_sp_angle['sp_angle']

    df_sp_row_id = df_sp[['id','row_id']].drop_duplicates()
    df_sp = df_sp[['id', 'HR_before', 'HR_after', 'sp_distance', 'sp_angle']]

    # Group by 'category' and apply aggregation functions
    df_sp_sum = df_sp.groupby('id').agg({
        'sp_distance': 'sum',    # Sum the values in 'values_1'
        'sp_angle': 'sum'    # Calculate the mean of 'values_2'
    })

    # Group by 'category' and apply aggregation functions
    df_sp_average = df_sp.groupby('id').agg({
        'sp_distance': 'mean',    # Sum the values in 'values_1'
        'sp_angle': 'mean'    # Calculate the mean of 'values_2'
    })

    # Group by 'category' and apply aggregation functions
    df_sp_median = df_sp.groupby('id').agg({
        'sp_distance': 'median',    # Sum the values in 'values_1'
        'sp_angle': 'median'    # Calculate the mean of 'values_2'
    })

    df_sp = df_sp[['id', 'HR_before', 'HR_after']]
    df_sp = df_sp.groupby('id').agg('mean')

    df_sp['sp_total_distance'] = df_sp_sum['sp_distance']
    df_sp['sp_average_distance'] = df_sp_average['sp_distance']
    df_sp['sp_median_distance'] = df_sp_median['sp_distance']

    df_sp['sp_total_angle'] = df_sp_sum['sp_angle']
    df_sp['sp_average_angle'] = df_sp_average['sp_angle']
    df_sp['sp_median_angle'] = df_sp_median['sp_angle']

    df_sp = df_sp.reset_index()
    df_sp = pd.merge(df_sp, df_sp_row_id, on='id', how='inner')

    # Saccadic Eye Movement
    # Convert the Unix timestamp to a normal timestamp (assuming it's in microseconds)
    df_sem['Timestamp'] = df_sem['Timestamp'] //100  # Convert from microseconds to seconds
    df_sem['datetime_timestamp'] = pd.to_datetime(df_sem['Timestamp'], unit='ms', utc=True)  # Convert to datetime

    # Filter the moving target to keep only rows with max or min values 
    max_value = df_sem['MovingTarget_X'].max()
    min_value = df_sem['MovingTarget_X'].min()

    df_sem = df_sem[(df_sem['MovingTarget_X'] == max_value) | (df_sem['MovingTarget_X'] == min_value)]

    # Create a discrete signal
    df_sem['MovingTarget_X_discrete'] = np.where(df_sem['MovingTarget_X'] == max_value, max_value, 
                                np.where(df_sem['MovingTarget_X'] == min_value, min_value, np.nan))

    df_sem['MovingTarget_X_discrete'] = df_sem['MovingTarget_X_discrete'].replace(max_value, 'right')
    df_sem['MovingTarget_X_discrete'] = df_sem['MovingTarget_X_discrete'].replace(min_value, 'left')

    def check_if_eye_is_inside_the_circle(MovingTarget_X, MovingTarget_Y, EyeTracker_X, EyeTracker_Y, circle_radius):
        position = (EyeTracker_X - MovingTarget_X)**2 + (EyeTracker_Y - MovingTarget_Y)**2
        if position <= circle_radius**2:
            return True
        else:
            return False
        
    df_sem_row_id = df_sem[['id','row_id']].drop_duplicates()
    all_latency = []
    discrete = ''
    counting = False
    start_time_change = 0
    circle_radius = 20

    for id in df_sem.id.unique():
        latency = []
        df_sem_id = df_sem[df_sem['id'] == id ]
        for row in df_sem_id.itertuples():
            # Check if the previous discrete is changing
            if discrete != row.MovingTarget_X_discrete:
                discrete = row.MovingTarget_X_discrete
                counting = True
                start_time_change = row.datetime_timestamp

            # If counting and eye is inside the circle, calculate latency
            if counting and check_if_eye_is_inside_the_circle(row.MovingTarget_X, row.MovingTarget_Y, row.EyeTracker_X, row.EyeTracker_Y, circle_radius):
                latency_delta = row.datetime_timestamp - start_time_change  # This is a timedelta object
                latency_in_milliseconds = latency_delta.total_seconds() * 1000  # Convert timedelta to milliseconds
                latency.append(latency_in_milliseconds)  # Append latency in milliseconds
                counting = False  # Stop counting after latency is recorded
                start_time_change = 0  # Reset start time

        all_latency.append(latency)

    sum_latency_list = []
    mean_latency_list = []
    median_latency_list = []

    for latency in all_latency:
        sum_latency_list.append(np.sum(latency))
        mean_latency_list.append(np.mean(latency))
        median_latency_list.append(np.median(latency))

    df_latency = pd.DataFrame()
    df_latency['id'] = df_sem.id.unique()
    df_latency['total_latency'] = sum_latency_list
    df_latency['mean_latency'] = mean_latency_list
    df_latency['median_latency'] = median_latency_list

    df_latency = pd.merge(df_latency, df_sem_row_id, on='id', how='left')

    df_sp = df_sp.reset_index()
    df_engineered = pd.merge(df_sp, df_latency, left_on='row_id', right_on='row_id',how='inner')

    df_engineered = df_engineered.drop(['index', 'id_x', 'id_y'], axis = 1)

    def count_vo2max_male(row, age, height, weight):
        return (70.597 - 0.246*age + 0.077*height - 0.222*weight - 0.147*row['HR_after'] )

    def count_vo2max_female(row, age, height, weight):
        return (70.597 -  0.185*age + 0.097*height - 0.246*weight - 0.122*row['HR_after'] )

    
    df_engineered['vo2_max'] = df_engineered.apply(lambda row: count_vo2max_male(row, age, height, weight) if row['sex'] == 1 else count_vo2max_female(row, age, height, weight), axis=1)
    
    # Feature Selection
    df_engineered = df_engineered.dropna()

    df_engineered['HR_before'] = HR_before

    df_engineered = df_engineered[['HR_before','sp_median_distance','sp_median_angle','median_latency']]

    # Inference
    model = joblib.load(MODEL_PATH)

    prediction = model.predict(df_engineered.iloc[0:1])

    print("Prediction:", prediction[0])

    return prediction[0]

def classify_vo2max(vo2max, sex):
    """
    Load Eyetracking and Heartrate CSV data and predict the VO2MAX from both data

    Parameters:
    ----------
    vo2max : float
        Directory Path of Eyetracking CSV file
    sex : bool
        Path of Heartrate CSV file

    Returns:
    -------
    predictions : int 
        The classification of fitness based on VO2MAX and sex value.
        Label:
        0 -> Low Fitness
        1 -> Average Fitness
        2 -> Good Fitness

    """
    if sex == 1:
        if vo2max < 37:
            return 0
        elif vo2max > 37 and vo2max < 52:
            return 1
        else:
            return 2
        
    elif sex == 0:
        if vo2max < 33:
            return 0
        elif vo2max > 33 and vo2max < 47:
            return 1
        else:
            return 2

predict_vo2max(EYETRACKING_DATA_PATH, HEARTRATE_DATA_PATH, MODEL_PATH )