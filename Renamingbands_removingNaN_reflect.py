# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:35:21 2024

@author: harsha
"""
import pandas as pd
import glob
import os

# Folder path where all reflectance CSV files are stored
reflectance_folder_path = 'C:/Users/harsha/.spyder-py3/Prismareflectance_LUCAS/'
reflectance_files = glob.glob(reflectance_folder_path + '*.csv')

# Folder path to save the cleaned CSV files
output_folder = 'C:/Users/harsha/.spyder-py3/Prismareflectance_LUCAS/Output_prisma_LUCAS_updated/'


# Function to rename reflectance bands consistently (band1, band2, ..., band234)
def rename_reflectance_columns(df):
    # Identify reflectance columns (assuming they contain 'Output VNIR/SWIR Cube raster layer')
    reflectance_columns = [col for col in df.columns if 'Output VNIR/SWIR Cube raster layer' in col]
    
    # Create a mapping to rename columns from "Output VNIR/SWIR Cube raster layer_XXX_X" to "bandX"
    rename_mapping = {old_col: f'band{i+1}' for i, old_col in enumerate(reflectance_columns)}
    
    # Rename the columns
    df.rename(columns=rename_mapping, inplace=True)
    return df

# Iterate over each reflectance CSV file, rename the columns, and save the file
for file in reflectance_files:
    #  Load the CSV file into a DataFrame
    df = pd.read_csv(file)
    
    #  Rename the reflectance columns consistently
    df = rename_reflectance_columns(df)
    
    #  Save the updated DataFrame back to the file (or a new location)
    df.to_csv(file, index=False)
    
    #  Remove rows where all band columns have missing (NaN or empty) values
    band_columns = [col for col in df.columns if col.startswith('band')]  # Get all band columns
    df_cleaned = df.dropna(subset=band_columns, how='all')  # Remove rows where all bands are NaN
    
    #  Save the cleaned DataFrame to the new folder
    filename = os.path.basename(file)  # Get the file name
    new_file_path = os.path.join(output_folder, filename)  # Construct the path to save the file
    df_cleaned.to_csv(new_file_path, index=False)  # Save the cleaned file
    
    print(f"Cleaned file saved: {new_file_path}")

