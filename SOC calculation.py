# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:37:17 2024

@author: harsha
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the two CSV files
file1 = pd.read_csv('SOC_calc_LUCAS.csv')
file2 = pd.read_csv('BulkDensity_2018.csv')

# Select the column to use for comparison (e.g., 'POINTID')
file1_column = file1[['POINTID']]  # Extract POINTID from file1
file2_column = file2[['POINTID']]  # Extract POINTID from file2

# Merge on 'POINTID' to get the common rows
common_rows = pd.merge(file1, file2, on='POINTID', how='outer')

# Filter the DataFrame to keep only the rows where the second column (OC) is not NaN
df_filtered = common_rows[common_rows['OC'].notna()]

# Replace commas with full stops in the second column (OC)
df_filtered['OC'] = df_filtered['OC'].str.replace(',', '.').astype(float)

columns_to_remove = ['BD 0-10', 'BD 10-20', 'BD 20-30' ]
df_filtered = df_filtered.drop(columns=columns_to_remove)

# Convert the 'OC(g/kg)' column to percentage

df_filtered['OC_percentage'] = df_filtered['OC'] / 10

# Optionally, remove the original 'OC(g/kg)' column
df_filtered = df_filtered.drop(columns=['OC'])

# Define the formula to calculate BD from OC from Manrique and Jones (1991)
def calculate_bd_from_oc(OC_percentage):
    # Add a small constant to OC to avoid log(0) which is undefined
    return 1.66 - 0.6 * np.log(OC_percentage + 1e-9)

# Step 3: Apply the formula to fill missing BD values
# Select rows with missing BD values
missing_bd_mask = df_filtered['BD 0-20'].isna()

# Calculate the missing BD values using the formula
df_filtered.loc[missing_bd_mask, 'BD 0-20'] = df_filtered.loc[missing_bd_mask, 'OC_percentage'].apply(calculate_bd_from_oc)




# Calculate mean (mu_oc and mu_bd)
mu_oc = df_filtered['OC_percentage'].mean()     # Mean of Organic Carbon (OC)
mu_bd = df_filtered['BD 0-20'].mean()     # Mean of Bulk Density (BD)

# Calculate standard deviation (sigma_oc and sigma_bd)
sigma_oc = df_filtered['OC_percentage'].std()   # Standard deviation of Organic Carbon (OC)
sigma_bd = df_filtered['BD 0-20'].std()   # Standard deviation of Bulk Density (BD)

# Calculate the covariance manually using the formula
cov_oc_bd = np.mean((df_filtered['OC_percentage'] - mu_oc) * (df_filtered['BD 0-20'] - mu_bd))

# Display the result
print(f"Covariance between OC and BD: {cov_oc_bd}")

# Print results
print(f"Mean OC_percentage (mu_oc): {mu_oc:.8f}")
print(f"Standard Deviation OC_percentage (sigma_oc): {sigma_oc:.8f}")
print(f"Mean BD 0-20 (mu_bd): {mu_bd:.8f}")
print(f"Standard Deviation BD 0-20 (sigma_bd): {sigma_bd:.8f}")
print(f"Covariance OC-BD (cov_oc_bd): {cov_oc_bd:.8f}")

depth_upper = 0.2  # Depth upper in meters
depth_lower = 0.0  # Depth lower in meters
# Depth difference
depth_diff = depth_upper - depth_lower

# Formula for SOC calculation
SOC_mean = (mu_oc/100) * mu_bd * (depth_lower - depth_upper) * 10


def calculate_sd_soc(mu_oc, sigma_oc, mu_bd, sigma_bd, cov_oc_bd, depth_upper, depth_lower):
    

    # SOC Standard Deviation (SOC_sd) formula components
    term1 = ((mu_oc/100)**2) * (sigma_bd**2)
    term2 = (mu_bd**2) * ((sigma_oc/100)**2)
    term3 = 2 * ((mu_oc/100) * mu_bd * cov_oc_bd)
    term4 = ((sigma_oc/100)**2) * (sigma_bd**2)
    term5 = cov_oc_bd**2
    
    SOC_sd = np.sqrt(term1 + term2 + term3 + term4 + term5) * depth_diff * 10
    return SOC_sd

# Calculate SOC Standard Deviation (SD_SOC) in Mg/ha
SOC_sd = calculate_sd_soc(mu_oc, sigma_oc, mu_bd, sigma_bd, cov_oc_bd, depth_upper, depth_lower)


# Calculate SOC stocks (SOC_Stock_Mg_ha) in Mg/ha
df_filtered['SOC_Stock_Mg_ha'] = df_filtered['BD 0-20'] * (df_filtered['OC_percentage'] / 100) * depth_diff * 10


#  Finally, the SOC stock might be reported for each row (as an example, using a formula like SOC = BD * OC)
df_filtered['SOC_stock_upper'] = df_filtered['SOC_Stock_Mg_ha'] + SOC_sd
df_filtered['SOC_stock_lower'] = df_filtered['SOC_Stock_Mg_ha'] - SOC_sd  


# Display results
print(f"Mean SOC Stock: {SOC_mean:.2f} Mg/ha")
#print("SOC Stock Range: {:.2f} to {:.2f} Mg/ha".format(SOC_stock_lower, SOC_stock_upper))

# View the filtered result
print(df_filtered)

# Save the common rows to a new CSV file
# df_filtered.to_csv('SOC_calc_LUCAS_2018.csv', index=False)

print(f"Standard deviation of SOC Stock: {SOC_sd :.8f} Mg/ha")
SOC_median = df_filtered['SOC_Stock_Mg_ha'].median()
SOC_updatedmean = df_filtered['SOC_Stock_Mg_ha'].mean()
print(f"Median SOC Stock: {SOC_median:.8f} Mg/ha")
print(f"Updated mean SOC Stock: {SOC_updatedmean:.8f} Mg/ha")

# Density plots
plt.figure(figsize=(12,6))

# Density plot to visualize the distribution of SOC values
plt.subplot(1, 2, 1) 
sns.kdeplot(df_filtered['SOC_Stock_Mg_ha'], fill=True, color='blue', alpha=0.5) 
plt.title('Density Plot of SOC Values') 
plt.xlabel('SOC (Mg/ha)') 
plt.ylabel('Density') 

# Density plot to visualize the distribution of range of SOC values considering SD  
plt.subplot(1, 2, 2) 
sns.kdeplot(df_filtered['SOC_stock_lower'], fill=True, color='orange', alpha=0.5, label='SOC stock Lower') 
sns.kdeplot(df_filtered['SOC_stock_upper'], fill=True, color='green', alpha=0.5, label='SOC stock Upper') 
plt.title('Density Plot of SOC Standard Deviation Ranges') 
plt.xlabel('Standard Deviation Range (Mg/ha)') 
plt.ylabel('Density') 
plt.legend() 

plt.tight_layout() 
plt.show()  


# Values for OC, BD, and SOC
oc_values = {
    "Mean of OC in % ": 2.67640719,
    "Standard Deviation of OC in %": 2.20259546
}

bd_values = {
    "Mean of BD in Mg/cubic.mts": 1.09554568,
    "Standard Deviation of BD in Mg/cubic.mts": 0.38800499
}

soc_values = {
    "Standard Deviation of SOC in Mg/ha ": 1.23656246,
    "Median of SOC in Mg/ha": 0.04546651,
    "Mean of SOC in Mg/ha": 0.04568891,
    "Covariance between OC and BD": -0.6476805917036427
}

# Prepare data for plotting
labels = list(oc_values.keys()) + list(bd_values.keys()) + list(soc_values.keys())
values = list(oc_values.values()) + list(bd_values.values()) + list(soc_values.values())

# Create groups for bar positions
n_oc = len(oc_values)
n_bd = len(bd_values)
n_soc = len(soc_values)
x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars

# Create the plot
plt.figure(figsize=(12, 6))

# Plot each group
plt.bar(x[:n_oc], list(oc_values.values()), width, label='Organic Content (OC in %) Values', color='skyblue')
plt.bar(x[n_oc:n_oc+n_bd], list(bd_values.values()), width, label='Bulk Density (BD in Mg/cubic.mts) Values', color='lightgreen')
plt.bar(x[n_oc+n_bd:], list(soc_values.values()), width, label='Soil Organic Carbon (SOC in Mg/ha) Values', color='salmon')

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.xlabel('Statistics of Organic Content (OC in %), Bulk Density (BD in Mg/cubic.mts), Soil Organic Carbon (SOC in Mg/ha)')
plt.ylabel('Integer values')
plt.title('Summary of Statistics of the LUCAS ground points in the Area of Interest in 0-20cm depth of soil surface')
plt.xticks(x, labels, rotation=45, ha='right')
plt.legend()

# Add gridlines
plt.grid(axis='y')

# Save the plot as a PNG image
plt.tight_layout()  # Adjust layout to make room for rotated labels
plt.savefig('summary_statistics_groundParameters.png', bbox_inches='tight')
plt.close()

print("PNG saved as 'summary_statistics_groundParameters.png'")
