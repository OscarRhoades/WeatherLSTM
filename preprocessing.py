import visualize

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder


def generate_synthetic_t_distribution(dataframe, df=3):
    # Get columns with missing values
    columns_with_missing = dataframe.columns[dataframe.isnull().any()].tolist()

    for col in columns_with_missing:
        # Compute means and std of column
        mean_val = dataframe[col].mean(skipna=True)
        std_val = dataframe[col].std(skipna=True)

        # Compute number of NaNs in column
        num_missing = dataframe[col].isnull().sum()

        # assumption of independence
        synthetic_values = np.random.standard_t(
            dataframe[col].size - num_missing, size=num_missing) * std_val + mean_val

        # set random student values to NaN values
        dataframe.loc[dataframe[col].isnull(), col] = synthetic_values

    return dataframe


def xgboost_impute(dataframe, synthetic_dataframe):
    columns_with_missing = dataframe.columns[dataframe.isnull().any()].tolist()

    # Find missing values
    for col in columns_with_missing:
        if synthetic_dataframe[col].isnull().any():
            raise ValueError(
                f"Column '{col}' in the synthetic DataFrame contains missing values.")

        # drop column to be adjusted
        df_train = synthetic_dataframe.dropna(subset=[col])

        # Fit xgboost regressor to all other columns
        X_train = df_train.drop(columns=col)
        y_train = df_train[col]
        model = XGBRegressor()
        model.fit(X_train, y_train)

        
        # Selecting missing values in colmn
        missing_indices = dataframe[dataframe[col].isnull()].index
        # infer with xgboost regressor
        X_missing = dataframe.loc[missing_indices].drop(columns=col)

        missing_predictions = model.predict(X_missing)
    
        dataframe.loc[missing_indices, col] = missing_predictions

    return dataframe


def preprocess_data(df):
    # Dropping the first column
    # df = df.drop(df.columns[0], axis=1)
    # changes date to index
    df["Date"] = df.index
    # map categorical directections to angles
    def direction_to_angle(direction):
        directions = {
            'N': 0,
            'NNE': 22.5,
            'NE': 45,
            'ENE': 67.5,
            'E': 90,
            'ESE': 112.5,
            'SE': 135,
            'SSE': 157.5,
            'S': 180,
            'SSW': 202.5,
            'SW': 225,
            'WSW': 247.5,
            'W': 270,
            'WNW': 292.5,
            'NW': 315,
            'NNW': 337.5
        }
        return directions.get(direction, None)

    df['WindGustDir'] = df['WindGustDir'].apply(
        direction_to_angle).astype('float64')
    df['WindDir9am'] = df['WindDir9am'].apply(
        direction_to_angle).astype('float64')

    # Select categorical variables
    object_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Change object variables to one-hot encodings
    label_encoder = LabelEncoder()
    for col in object_columns:
        df[col] = label_encoder.fit_transform(df[col])

    # typecast to float
    int_columns = df.select_dtypes(include=['int64']).columns
    df[int_columns] = df[int_columns].astype(float)
    # convert values to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    non_float_nan_mask = ~df.applymap(
        lambda x: isinstance(x, float) or np.isnan(x))
    df[non_float_nan_mask] = np.nan

    return df


def normalize(df, ignore):
    # Transforms data to be between zero and 0 1 for all columns
    normalized_df = df.copy()
    for column in normalized_df.columns:
        if column != ignore:
            col_min = normalized_df[column].min()
            col_max = normalized_df[column].max()
            normalized_df[column] = (
                normalized_df[column] - col_min) / (col_max - col_min)
    return normalized_df


def percentage_of_nans_per_column(dataframe):
    total_values = len(dataframe)
    nan_counts = dataframe.isnull().sum()

    print("Percentage of NaNs in each column:")
    for column, count in nan_counts.items():
        percentage = (count / total_values) * 100
        print(
            f" {column} | Percentage: {percentage:.2f}% | NaNs: {count} | Total: {total_values:.2f}")


def preprocess_synthetic(file):
   
    df = pd.read_csv(file)

    #imputes and gets data into 
    df_numeric = preprocess_data(df.copy())

    def fix_target(df, column):
        df_copy = df.copy()
        df_copy[column] = df_copy[column].apply(lambda x: 0 if x > 1 else x)
        return df_copy  
    

    df_numeric = fix_target(df_numeric, "RainTomorrow")
    # removes rows with NaN target, I am not sure this actually helps
    nan_indices = df_numeric.index[df_numeric["RainTomorrow"].isnull()]
    df_numeric = df_numeric.drop(nan_indices, axis=0)

    nan_indices = df_numeric.index[df_numeric["RainToday"].isnull()]
    df_numeric = df_numeric.drop(nan_indices, axis=0)

    # visualize data and show NaN percentages by column
    visualize.compute_correlation_and_plot_all(df_numeric, "RainTomorrow")
    percentage_of_nans_per_column(df)

    # generates the rand-t distributions values
    df_with_synthetic_t = generate_synthetic_t_distribution(
        df_numeric.copy(), df=df_numeric.size)

    # adjust missing values with XGboost
    df_imputed_with_xgboost = xgboost_impute(
        df_numeric.copy(), df_with_synthetic_t)
    # # print(df_with_synthetic_t)
    # # print(df_imputed_with_xgboost)

    def cap_values_above(df, column):
        df_copy = df.copy()
        df_copy[column] = df_copy[column].apply(lambda x: 1 if x > 1 else x)
        return df_copy    


    # normalize values
    normalized_df = normalize(df_imputed_with_xgboost, ignore = "RainTomorrow")

    normalized_df = cap_values_above(normalized_df,"RainTomorrow")
    normalized_df.to_csv('synthetic_data.csv', index=False)

    return normalized_df
