### K MEANS BRAIN SCAN USING PYTHON AND ML ####
### source: C:\Users\posit\Documents\Training\1x10\Recap250531\Projects\06-K Means Brain Scan-Python\Brain_Scan_Labelled.png
################################################
# pwd -> C:\Users\posit

import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image 


root=r'Documents\Training\1x10\Recap250531\Projects'
folder='06-K Means Brain Scan-Python'
filename= 'Brain_Scan_Labelled.png'
filepath= f'{root}\\{folder}\\{filename}' # or use below
filepath = rf'{root}\{folder}\{filename}'

# or use os.path.join for better cross-platform compatibility
if not os.path.exists(filepath):
    print("File not found:", filepath)
    sys.exit(1)  # Exit the script if the file does not exist

print('done with file path')
# Read the png file into a DataFrame    
img = Image.open(filepath)
img=img.resize((200, 200))  # Resize the image to reduce processing time
img_array = np.array(img)

# display the original image
plt.figure(figsize=(10, 10))
plt.imshow(img_array)
plt.axis('off')  # Hide the axes
plt.title('Original Brain Scan Image')
plt.show()

# Define the nb of clusters
k = 4  # Number of clusters for KMeans
# Reshape the image array to a 2D array of pixels
h,w,c = img_array.shape
pixels = img_array.reshape(h*w, c)  # Reshape to (num_pixels, num_channels)
# Perform KMeans clustering
kmeans = KMeans(n_clusters=k, n_init=10)
labels = kmeans.fit_predict(pixels)

# Create a new image with the clustered colors
clustered_image = kmeans.cluster_centers_[labels].reshape(h, w, c).astype(int) # 3D

# Display the clustered image
plt.figure(figsize=(10, 10))
plt.imshow(clustered_image)
plt.axis('off')  # Hide the axes
plt.title(f'Clustered Brain Scan Image with {k} Clusters')
plt.show()

# Elbow method to find the optimal number of clusters
sse=[]  # Sum of squared errors
K=range(2,11)  # Range of K values to test 2:10
for i in K:
    kmeans_elbow = KMeans(n_clusters=i, n_init=10, random_state=42)
    kmeans_elbow.fit(pixels)
    sse.append(kmeans_elbow.inertia_)  # Inertia is the sum of squared distances to closest cluster center

# Plot the elbow method
plt.figure(figsize=(8, 6))
plt.plot(K, sse, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.xticks(K)
plt.grid(True)
plt.show()

# Plot each section of the clustered image
k=4
for i in range(k):
    cluster_mask=(labels.reshape(h,w)==i) # Create a mask for the current cluster
    section = img_array * cluster_mask  # does not run here img_array is 3D, cluster_mask is 2D - img _array should be the clustered image

    plt.figure()
    plt.imshow(section,cmap=plt.cm.gray)
    plt.axis('off')  # Hide the axes
    plt.title(f'Cluster {i+1}')
    plt.show()


if 0:
    # Display the inertia values
    print("Inertia values for K from 1 to 10:")
    for i, value in enumerate(sse, start=1):
        print(f"K={i}: Inertia={value:.2f}")
    # Display the cluster centers
    print("Cluster centers (RGB values):")
    for i, center in enumerate(kmeans.cluster_centers_, start=1):
        print(f"Cluster {i}: {center.astype(int)}")  # Convert to int for display
    # Display the labels for the original image
    print("Labels for the original image:")
    print(labels[:100])  # Display first 100 labels for brevity

    # Display the clustered image as a PIL image
    clustered_image_pil = Image.fromarray(clustered_image.astype('uint8'))
    # Display the clustered image
    plt.figure(figsize=(10, 10))
    plt.imshow(clustered_image_pil)
    plt.axis('off')  # Hide the axes
    plt.title(f'Clustered Brain Scan Image with {k} Clusters (PIL)')
    plt.show()

if 0:
    ##################### DICOM IMAGE PROCESSING ######################
    # import lib for dcm image
    import pydicom
    # Load a DICOM file
    dicom_path='Documents/Training/1x10/Recap250531/Projects/06-K Means Brain Scan-Python/brain_scan.dcm'
    dicom_image = pydicom.dcmread(dicom_path)
    # Convert the DICOM pixel data to a NumPy array
    dicom_array = dicom_image.pixel_array
    #dicom_array=img_array
    # Normalize the pixel values to the range [0, 255]
    dicom_array = (dicom_array - np.min(dicom_array)) / (np.max(dicom_array) - np.min(dicom_array)) * 255
    # Display the DICOM image
    plt.figure(figsize=(10, 10))
    plt.imshow(dicom_array, cmap='gray')
    plt.title('DICOM Brain Scan Image')
    plt.show()

    # Reshape the DICOM image array to a 2D array of pixels
    h,w = dicom_array.shape
    pixels_dicom = dicom_array.reshape(h*w, 1)  # Reshape to (num_pixels, num_channels)
    # Perform KMeans clustering on the DICOM image
    kmeans_dicom = KMeans(n_clusters=k, random_state=42,n_init=10)
    labels_dicom = kmeans_dicom.fit_predict(pixels_dicom)
    # Create a new image with the clustered colors for DICOM
    clustered_image_dicom = kmeans_dicom.cluster_centers_[labels_dicom].reshape(h, w).astype(int)
    # Display the clustered DICOM image
    plt.figure(figsize=(10, 10))
    plt.imshow(clustered_image_dicom, cmap='gray')
    plt.axis('off')  # Hide the axes
    plt.title(f'Clustered DICOM Brain Scan Image with {k} Clusters')
    plt.show()

    # Elbow method to find the optimal number of clusters
    inertia = []
    for i in range(1, 11):
        kmeans_temp = KMeans(n_clusters=i, n_init=10, random_state=42)
        kmeans_temp.fit(pixels)
        inertia.append(kmeans_temp.inertia_)
    # Plot the elbow method
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 11))
    plt.grid()
    plt.show()
    # Display the inertia values
    print("Inertia values for K from 1 to 10:")
    for i, value in enumerate(inertia, start=1):
        print(f"K={i}: Inertia={value:.2f}")
    # Display the cluster centers
    print("Cluster centers (RGB values):")
    for i, center in enumerate(kmeans.cluster_centers_, start=1):
        print(f"Cluster {i}: {center.astype(int)}")  # Convert to int for display
    # Display the cluster centers for DICOM
    print("Cluster centers for DICOM (grayscale values):")
    for i, center in enumerate(kmeans_dicom.cluster_centers_, start=1):
        print(f"Cluster {i}: {center[0]:.2f}")  # Display grayscale value
    # Display the labels for the original image
    print("Labels for the original image:")
    print(labels[:100])  # Display first 100 labels for brevity
    # Display the labels for the DICOM image
    print("Labels for the DICOM image:")
    print(labels_dicom[:100])  # Display first 100 labels for brevity
    # Save the clustered images
    clustered_image_pil = Image.fromarray(clustered_image.astype('uint8'))

