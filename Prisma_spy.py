# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:38:14 2024

@author: harsha
"""
import h5py
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import math
from rasterio.transform import from_origin

  
 # dimensions 
num_rows = 100
num_cols = 100
num_bands = 239

hyperspectral_data = np.random.rand(num_rows, num_cols, num_bands)

wavelengths = np.linspace(400, 2505, num_bands) # wavelengths in nanometers
file_path = 'hyperspectral_data.h5'
with h5py.File(file_path, 'w') as file:
    file.create_dataset('hyperspectral_data', data = hyperspectral_data)
    file.create_dataset('wavelengths', data = wavelengths)
    file['hyperspectral_data'].attrs['description'] = 'Synthetic hyperspectral data'
    file['wavelengths'].attrs['description'] = 'Wavelengths corresponding to hyperspectral bands'
    
    
print(f"HDF5 file '{file_path}'created successfully")

with h5py.File(file_path, 'r') as file:
    hyperspectral_data = file['hyperspectral_data'][()]
    wavelengths = file['wavelengths'][()]
    print(f"hyperspectral_data shape : {hyperspectral_data.shape}")
    print(f"wavelengths shape : {wavelengths.shape}")
    print(f"hyperspectral_data description : {file['hyperspectral_data'].attrs['description']}")
    print(f"wavelengths description : {file['wavelengths'].attrs['description']}")
    
    
   
plt.figure(figsize=(10,4))
plt.plot(wavelengths, marker = 'o')
plt.xlabel('Band Number')
plt.ylabel('Wavelength (nm)')
plt.title('Synthetic Hyperspectral Bands')
plt.grid(True)
plt.show()

file_path = 'PRS_L2D_STD_20210923103054_20210923103059_0001.he5'

with h5py.File(file_path, 'r') as file:
    def print_structure(name, obj):
        print(name)
        
    file.visititems(print_structure) 
    
file = h5py.File(file_path, 'r')
# SWIR channel : 920-2505 nm
SWIR = file['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Data Fields']['SWIR_Cube']
# VNIR channel : 400-1010 nm
VNIR = file['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Data Fields']['VNIR_Cube']
reflectance_data = np.concatenate((VNIR, SWIR), axis=1)

print("SWIR data shape :", SWIR.shape)
print("VNIR data shape :", VNIR.shape)

# Extract RGB bands 
Red_band = VNIR[:, 43, :]
Green_band = VNIR[:, 26, :]
Blue_band = VNIR[:, 14, :]

rgb_image1 = np.stack([Red_band, Green_band, Blue_band], axis=-1)

# calculate the mean reflectance spectrum across all the pixels
mean_spectrum = np.mean(reflectance_data, axis=(0,2))

# plot reflectance vs wavelength of that pixel 
plt.figure(figsize = (10,6))
plt.plot(wavelengths, mean_spectrum, label = f'Mean Reflectance spectrum')
plt.xlabel('wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Mean Reflectance spectrum across VNIR and SWIR bands')
plt.grid(True)
plt.legend()
plt.show()



file = h5py.File(file_path, 'r')
# extracting the geo-coordiates (Lat and Long arrays)
Lat = file['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Geolocation Fields']['Latitude']

Long = file['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Geolocation Fields']['Longitude']
# extractig the acquisition time of the data
Time_acq = file['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Geolocation Fields']['Time']


# viewing the subset of lat and long values of the file, since the arrays are large

print("Latitude shape:", Lat.shape)
print("Longitude shape:", Long.shape)
print("Acquisition Time shape:", Time_acq.shape)

print("Sample Latitude (first 5 values):", Lat[:5, :5])
print("Sample Longitude (first 5 values):", Long[:5, :5])
print("Sample Acquisition Time (first 5 values):", Time_acq[:5, :5])

# finding the spatial resolution in meters of the he5 file 
upper_left_lat = Lat[0, 0]
upper_left_long = Long[0, 0]

lower_right_lat = Lat[-1, -1]
lower_right_long = Long[-1, -1]

width_map_units =  upper_left_lat - lower_right_lat # latitude extent
height_map_units = lower_right_long - upper_left_long # longitude extent

pixel_size_lat = width_map_units/1224
pixel_size_long = height_map_units/1256

central_lat = (upper_left_lat + lower_right_lat)/2
long_conversion_factor = 111*math.cos(math.radians(central_lat))

pixel_size_lat_m = pixel_size_lat*111000 #  meters
pixel_size_long_m = pixel_size_long*long_conversion_factor*1000 #  meters

print(f"Latitude pixel size: {pixel_size_lat_m} meters ")
print(f"Longitude pixel size: {pixel_size_long_m} meters ")



#CRS information 
if 'projection' in file['/HDFEOS INFORMATION'].attrs:
    crs_info = file['/HDFEOS INFORMATION'].attrs['projection']
    print("CRS Information:". crs_info)
else:
    print("No CRS Information found in attributes.")


# data properties 
height, width = reflectance_data.shape[0], reflectance_data.shape[2]
count = reflectance_data.shape[1] # number of bands
dtype = rasterio.float32

crs = 'EPSG:4326' 

transform = from_origin(upper_left_long, upper_left_lat, pixel_size_long, pixel_size_lat) # always should be in units of degrees for WGS84 CRS (EPSG:4326)

profile = {
    'driver': 'GTiff',
    'height': height,
    'width': width,
    'count': count,
    'dtype': dtype,
    'crs': crs,
    'transform': transform,
    'compress': 'lzw'
    }

# define the output file path
prisma_tiff_path = 'reflectance_data.tif' 
with rasterio.open(prisma_tiff_path, 'w', **profile) as dst:
    for i in range(reflectance_data.shape[1]):
        dst.write(reflectance_data[:, i, :], i+1) # write each band
    

    
print(f"Reflectance data tiff file over VNIR and SWIR bands is saved to {prisma_tiff_path}")



