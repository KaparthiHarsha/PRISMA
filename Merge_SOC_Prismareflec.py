# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:35:42 2024

@author: harsha
"""

import pandas as pd
import glob

#  Load the SOC File into a DataFrame
soc_file_path = 'C:/Users/harsha/.spyder-py3/SOC_calc_LUCAS_2018_updated.csv'  # Replace with the actual path to your SOC file
soc_df = pd.read_csv(soc_file_path)

# Assuming the reflectance files are in a folder and have a common pattern
reflectance_folder_path = 'C:/Users/harsha/.spyder-py3/Prismareflectance_LUCAS/Output_prisma_LUCAS_updated/'
reflectance_files = glob.glob(reflectance_folder_path + '*.csv')


# Initialize an empty DataFrame to store all reflectance data
combined_reflectance_df = pd.DataFrame()

# Load all reflectance CSV files and concatenate them into one DataFrame
for file in reflectance_files:
    # Load the reflectance file
    reflectance_df = pd.read_csv(file)
    
    # Append to the combined reflectance DataFrame
    combined_reflectance_df = pd.concat([combined_reflectance_df, reflectance_df])

# Merge the SOC DataFrame with the combined reflectance DataFrame on 'pointID'
# Use 'inner' join to keep only pointIDs that are present in both SOC and reflectance data
merged_df = pd.merge(soc_df, combined_reflectance_df, on='POINTID', how='inner')

# Save the merged DataFrame to a new CSV file
output_file_path = 'C:/Users/harsha/.spyder-py3/merged_soc_reflectance.csv'
merged_df.to_csv(output_file_path, index=False)

print(f"Merged file saved to {output_file_path}")