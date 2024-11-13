# The function provided below is used for inferencing the model
import os
import pandas as pd
import re
# from dotenv import load_dotenv
import numpy as np
import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, KFold
import joblib

# load_dotenv()

# EYETRACKING_DATA_PATH = os.getenv('EYETRACKING_DATA_PATH')
# HEARTRATE_DATA_PATH = os.getenv('HEARTRATE_DATA_PATH')
# MODEL_PATH = os.getenv('MODEL_PATH')

def load_data(EYETRACKING_DATA_PATH, HEARTRATE_DATA_PATH):
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

    return df_eyetracking_sem, df_eyetracking_sp, df_heartrate

def data_preprocessing_and_feature_engineering(df_sp, df_sem, df_heartrate):
    
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

    df_sp_angle['vector_e_length'] = df_sp_angle['vector_e_x']**2 + df_sp_angle['vector_e_y']**2
    df_sp_angle['vector_e_length'] = np.sqrt(df_sp_angle['vector_e_length'])

    df_sp_angle['vector_g_length'] = df_sp_angle['vector_g_x']**2 + df_sp_angle['vector_g_y']**2
    df_sp_angle['vector_g_length'] = np.sqrt(df_sp_angle['vector_g_length'])

    df_sp_angle['vector_e_dot_g'] = df_sp_angle['vector_e_x'] * df_sp_angle['vector_g_x'] + df_sp_angle['vector_e_y'] * df_sp_angle['vector_g_y']

    df_sp_angle['sp_angle'] = np.arccos(df_sp_angle['vector_e_dot_g'] / (df_sp_angle['vector_e_length'] * df_sp_angle['vector_g_length']))

    df_sp['sp_angle'] = df_sp_angle['sp_angle']

    df_sp_row_id = df_sp[['id','row_id']].drop_duplicates()
    df_sp = df_sp[['id', 'HR_before', 'HR_after', 'sp_distance', 'sp_angle','age','height','weight','sex']]

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

    df_sp = df_sp[['id', 'HR_before', 'HR_after','age','height','weight','sex']]
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

    def count_vo2max_male(row):
        return (70.597 - 0.246*row['age'] + 0.077*row['height'] - 0.222*row['weight'] - 0.147*row['HR_after'] )

    def count_vo2max_female(row):
        return (70.597 -  0.185*row['age'] + 0.097*row['height'] - 0.246*row['weight'] - 0.122*row['HR_after'] )


    df_engineered['vo2_max'] = df_engineered.apply(lambda row: count_vo2max_male(row) if row['sex'] == 1 else count_vo2max_female(row), axis=1)
    # Feature Selection
    df_engineered = df_engineered.dropna()

    df_engineered = df_engineered[['HR_before','vo2_max','sp_average_distance','sp_average_angle','mean_latency']]

    return df_engineered


def create_model(df, location_path):
    """
    Load Eyetracking and Heartrate CSV data and predict the VO2MAX from both data

    Parameters:
    ----------
    EYETRACKING_DATA_PATH : string
        Directory Path of Eyetracking CSV file
    HEARTRATE_DATA_PATH : string
        Path of Heartrate CSV file

    Returns:
    -------

    """
    # Assuming 'df' is your DataFrame and 'target_column' is the name of the target variable
    X = df.drop(columns=['vo2_max'])  # Features (drop the target column)
    y = df['vo2_max']                 # Target (the column you're predicting)

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_model(model, model_name, X, y):
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Perform 5-fold cross-validation
        cv_scores = cross_validate(model, X, y, cv=kfold, scoring=['neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_percentage_error'])

        # Extracting individual metrics from the results
        rmse_scores = -cv_scores['test_neg_root_mean_squared_error']  # Negate to make positive
        # r2_scores = cv_scores['test_r2']
        mape_scores = -cv_scores['test_neg_mean_absolute_percentage_error']  # Negate to make positive

        # return [model_name, round(float(rmse_scores.mean()),5), round(float(r2_scores.mean()),5), round(float(mape_scores.mean())*100,5)]
        return [model_name, round(float(rmse_scores.mean()),5), round(float(mape_scores.mean())*100,5)]
    
    rf = RandomForestRegressor(random_state=42)

    # rf.fit(X_train, y_train)

    # y_pred_lr = rf.predict(X_test)

    rf_scores = evaluate_model(rf, 'Random Forest Regressor', X, y)

    rf = RandomForestRegressor(random_state=42)

    # Retrain the model on the training data
    rf.fit(X, y)
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Assuming 'model' is your trained model
    joblib.dump(rf, f'{location_path}/random_forest_{timestamp}.pkl')

    print(rf_scores)



def main(EYETRACKING_DATA_PATH, HEARTRATE_DATA_PATH):
    # Preprocessing
    df_sem, df_sp, df_heartrate = load_data(EYETRACKING_DATA_PATH, HEARTRATE_DATA_PATH)

    df = data_preprocessing_and_feature_engineering(df_sp, df_sem, df_heartrate)

    create_model(df, './models')

main('data/raw/eyetracking', 'data/raw/heartrate/00_HR-data.csv')