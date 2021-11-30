import numpy as np
import pandas as pd

from src.data_preparation.preprocess_data import get_feature_target_array

from pathlib import Path

BASE_DATA_DIR = Path('data')

df_1 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set1_edited.xlsx", header=1)
df_2 = pd.read_csv(BASE_DATA_DIR/"Data_File_SW_prop_ML_set2.csv")
df_3 = pd.read_csv(BASE_DATA_DIR/"Data_File_SW_prop_ML_set3.csv")
df_4 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set4.xlsx", header=1)
df_5 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set5.xlsx", header=1)
df_6 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set6.xlsx", header=1)
df_7 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set7.xlsx", header=1)
df_8 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set8.xlsx", header=1)
df_9 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set9.xlsx", header=1)
df_10 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set10.xlsx", header=1)
df_11 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set11.xlsx", header=1)
df_12 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set12.xlsx", header=1)
df_13 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set13.xlsx", header=1)
df_14 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set14.xlsx", header=1)
df_15 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set15.xlsx", header=1)
df_16 = pd.read_excel(BASE_DATA_DIR/"Data_File_SW_prop_ML_set16.xlsx", header=1)

    
def create_dataset():
    """
    This function merges the feature and target array from all the dataframes.
    Returns
    -------
        Concatenated dataset in the form of an array
    """
    X = []
    y = []
    X_df1, y_df1 = get_feature_target_array(df_1)
    X_df2, y_df2 = get_feature_target_array(df_2)
    X_df3, y_df3 = get_feature_target_array(df_3)
    X_df4, y_df4 = get_feature_target_array(df_4)
    X_df5, y_df5 = get_feature_target_array(df_5)
    X_df6, y_df6 = get_feature_target_array(df_6)
    X_df7, y_df7 = get_feature_target_array(df_7)
    X_df8, y_df8 = get_feature_target_array(df_8)
    X_df9, y_df9 = get_feature_target_array(df_9)
    X_df10, y_df10 = get_feature_target_array(df_10)
    X_df11, y_df11 = get_feature_target_array(df_11)
    X_df12, y_df12 = get_feature_target_array(df_12)
    X_df13, y_df13 = get_feature_target_array(df_13)
    X_df14, y_df14 = get_feature_target_array(df_14)
    X_df15, y_df15 = get_feature_target_array(df_15)
    X_df16, y_df16 = get_feature_target_array(df_16)

    X = X_df1.tolist() + X_df2.tolist() + X_df3.tolist() + X_df4.tolist() + X_df5.tolist() + X_df6.tolist() + \
        X_df7.tolist() + X_df8.tolist() + X_df9.tolist() + X_df10.tolist() + X_df11.tolist() + X_df12.tolist() + \
            X_df13.tolist() + X_df14.tolist() + X_df15.tolist() + X_df16.tolist() 
    y = y_df1.tolist() + y_df2.tolist() + y_df3.tolist() + y_df4.tolist() + y_df5.tolist() + y_df6.tolist() + \
        y_df7.tolist() + y_df8.tolist() + y_df9.tolist() + y_df10.tolist() + y_df11.tolist() + y_df12.tolist() + \
            y_df13.tolist() + y_df14.tolist() + y_df15.tolist() + y_df16.tolist()
    X_arr = np.array(X).astype(np.float)
    y_arr = np.array(y).astype(np.float)
    return X_arr, y_arr 

def delete_rows_with_colname(df):
    """
    This function deletes the rows with column names.
    Returns:
    --------
        dataframe
    """
    df.drop(df.index[df['ACE_Bx'] == 'ACE_Bx'], inplace=True)
    df.drop(df.index[df['ACE_Bx'] == 'ACE_Bx '], inplace=True)
    df.drop(df.index[df['ACE_Bx'] == ' ACE_Bx'], inplace=True)
    df.drop(df.index[df['ACE_Bx'] == ' ACE_Bx '], inplace=True)
    return df

def get_all_concatenated_dataframes():
    """
    This function concatenates all dataframes. Is used for plotting attributes.
    Returns:
    --------
        Concatenated dataframe
    """
    df_1.columns = df_1.columns.str.replace(' ','')
    df_2.columns = df_2.columns.str.replace(' ','')
    df_3.columns = df_3.columns.str.replace(' ','')
    df_4.columns = df_4.columns.str.replace(' ','')
    df_5.columns = df_5.columns.str.replace(' ','')
    df_6.columns = df_6.columns.str.replace(' ','')
    df_7.columns = df_7.columns.str.replace(' ','')
    df_8.columns = df_8.columns.str.replace(' ','')
    df_9.columns = df_9.columns.str.replace(' ','')
    df_10.columns = df_10.columns.str.replace(' ','')
    df_11.columns = df_11.columns.str.replace(' ','')
    df_12.columns = df_12.columns.str.replace(' ','')
    df_13.columns = df_13.columns.str.replace(' ','')
    df_14.columns = df_14.columns.str.replace(' ','')
    df_15.columns = df_15.columns.str.replace(' ','')
    df_16.columns = df_16.columns.str.replace(' ','')
    df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10, df_11, df_12, df_13, df_14, df_15, df_16])
    df = delete_rows_with_colname(df)
    return df


def reshape_each_data_1_by_880():
    """
    This function reshapes 80 by 11 matrix to 1 by 880 vector.
    Returns
    -------
        A dataset consisting of an array reshaped to 1 by 880 and corresponding y values
    """
    X_reshaped = []
    X_arr, y_arr = create_dataset()
    for i in range(0, len(X_arr)):
        X_reshaped.append(X_arr[i].reshape(-1, 80*11))
    return np.array(X_reshaped), y_arr

def get_first_row_each_matrix():
    """
    This function gives only the first row of each matrix.
    Returns
    -------
       A dataset consisting of an array of shape 1 by 11 and corresponding y values 

    """
    X_first_row = []
    X_arr, y_arr = create_dataset()
    for i in range(0, len(X_arr)):
        X_first_row.append(X_arr[i][0])
    return np.array(X_first_row), y_arr

def get_average_of_columns_each_matrix():
    """
    This function computes the average of each column of the matrix and comprises a 1 by 11 vector
    Returns
    -------
       A dataset consisting of an array of shape 1 by 11 and corresponding y values 

    """
    X_average = []
    X_arr, y_arr = create_dataset()
    for i in range(0, len(X_arr)):
        X_average.append(np.mean(X_arr[i], axis = 0))
    return np.array(X_average), y_arr 
