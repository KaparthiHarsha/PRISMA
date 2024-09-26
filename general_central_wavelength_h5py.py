# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:24:03 2024

@author: harsha
"""
import h5py
import numpy as np
import pandas as pd

# Path to your HDF5 file
file_path = 'hyperspectral_data.h5'

# Open the HDF5 file and extract wavelengths
with h5py.File(file_path, 'r') as file:
    # Extract the wavelengths dataset
    wavelengths = file['wavelengths'][()]

# Generate the band numbers from 1 to 239
band_numbers = np.arange(1, len(wavelengths) + 1)

# Create a DataFrame with 'Band Number' and 'Central Wavelength'
df = pd.DataFrame({
    'Band Number': band_numbers,
    'Central Wavelength (nm)': wavelengths
})

# Save the DataFrame to a CSV file
output_csv_path = 'wavelengths_band_numbers.csv'
df.to_csv(output_csv_path, index=False)

print(f"CSV file saved to {output_csv_path}")

