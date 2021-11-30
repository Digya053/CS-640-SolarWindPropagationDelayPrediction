import numpy as np
import pandas as pd

from src.data_preparation.interpolate_data import *

def compute_new_row_number(df):
  """
  This function computes the row number where the new matrix starts.
  Parameters
  ----------
    df: DataFrame
      DataFrame whose row numbers are to be computed
  Returns
  -------
    List of row numbers 
  """
  new_row_number = df[df["Delay"] == "Delay"].index.values.tolist()
  if len(new_row_number) == 0:
    new_row_number = df[df["Delay"] == "Delay "].index.values.tolist()
  new_row_number.append(df.shape[0])
  return new_row_number

def create_data_array(new_row_number, df):
  """
  This function creates a new dataframe excluding the row number where column names are written
  Parameters:
  ----------
    new_row_number: List
      List of row number where column names are written
    df: DataFrame
      Dataframe which needs to be re-created excluding new_row_number
  Returns
  -------
    DataFrame excluding the column names
  """
  data_df = []
  data_df.append(df[0:new_row_number[0]].to_numpy())
  for index, n in enumerate(new_row_number):
    try:
      data_df.append(df[n+1:new_row_number[index+1]].to_numpy())
    except IndexError:
      break
  return data_df

def insert_row(row_number, df, row_value):
  """
  This function inserts the interpolated row in the dataframe.
  Parameters:
  -----------
    row_number: Integer
      The row number where interpolated data is to be stored
    df: DataFrame
      DataFrame where the insertion of row should be done
    row_value: list
      The interpolated row values
  Returns:
  --------
    DataFrame with values inserted

  """
  # Slice the upper half of the dataframe
  df1 = df[0:row_number]
  # Store the result of lower half of the dataframe
  df2 = df[row_number:]
  # Insert the row in the upper half dataframe
  df1.loc[row_number]=row_value
  # Concat the two dataframes
  df_result = pd.concat([df1, df2])
  # Reassign the index labels
  df_result.index = [*range(df_result.shape[0])]
  # Return the updated dataframe
  return df_result

def insert_all_new_records(df, new_row_list, row_values):
  """
  This function calls insert_row and inserts all new records in dataframe df
  Parameters:
  -----------
    df: DataFrame
      DataFrame where interpolated values are to be inserted
    new_row_list: List
      The row list where new values are to be inserted
    row_values: List
      The values which are to be inserted in the rows.
  Returns:
  --------
    DataFrame with all interpolated values inserted
  """
  for i in range(0, len(new_row_list)):
    row = np.append(row_values[i], 0)
    df = insert_row(new_row_list[i], df, row)
  return df

def get_feature_target_array(df):
  """
  This function returns a feature and target array from the dataframe
  Parameters
  ----------
    df: DataFrame
      DataFrame whose feature and target are to be extracted and converted to array
  Returns
  --------
    Arrays of feature and target
  """
  df.columns = df.columns.str.replace(' ','')
  new_row_number = compute_new_row_number(df)
  df['Delay'].fillna('', inplace=True)
  X_df = df.iloc[:, :-1]
  Y_df = df['Delay']
  data_df = create_data_array(new_row_number, X_df)
  if np.diff(new_row_number)[0] < 81:
    interpolated_row = find_linearly_interpolated_row(data_df)
    df = insert_all_new_records(df, new_row_number, interpolated_row)
    new_row = compute_new_row_number(df)
    X_df = df.iloc[:, :-1]
    Y_df = df['Delay']
    data_df = create_data_array(new_row, X_df)
  y_df = Y_df.to_numpy()
  y_df = [i for i in y_df if i != '' and i!='Delay ' and i!='Delay' and i!=0 and i!=np.NaN]
  return np.array(data_df), np.array(y_df)


