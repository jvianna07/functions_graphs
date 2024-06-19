from typing import Tuple
import os
from random import randint
import shutil
import pickle

import numpy as np
import cv2


def imageResizer(path_source: str, path_destination:str, size=(128,128), ext='.png'):
    '''
    Resize to a specific size.
    path_source - source folder path
    path_destination - destination folder path
    size= image size, default is 128x128
    ext = file extension, default is *.png
    '''

    for folder in os.listdir(path_source):
        currentFolder = os.path.join(path_source, folder)
        if os.path.isdir(currentFolder):

            # Create new folder in processed images
            new_folder  =  os.path.join(path_destination,folder)
            os.makedirs(new_folder, exist_ok = True)
           
            for item in os.listdir(currentFolder):
                f_name, f_ext = os.path.splitext(item)
                if f_ext in ['.png','.bmp']:
    
                    # Processar a imagem
                    image_path = os.path.join(currentFolder,item)
                    image = cv2.imread(image_path)
                    image = image.copy()
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    resized_img = cv2.resize(image,size, interpolation = cv2.INTER_AREA)
        
                    # Salvar a imagem
                    target_path = os.path.join(new_folder, f_name+ext)
                    cv2.imwrite(target_path, resized_img)

def datasetSpliter(path_source: str, path_destination: str, sample_size: str = 145):
    '''
    Sample some items from dataset
    path_source - source folder path
    path_destination - destination folder path
    sample_size = Number of items to be randomly selected in each folder, default is 145
    '''

    # loop trough folders 
    for folder in os.listdir(path_source):
        currentFolder = os.path.join(path_source, folder)
        if os.path.isdir(currentFolder):

            # Create new folder in processed images
            destination_folder  =  os.path.join(path_destination,folder)
            os.makedirs(destination_folder, exist_ok = True)

            # Select items randomly and copy to destination
            count = 0
            while count < sample_size:

                # List all current folder items 
                current_folder_imgs = os.listdir(currentFolder)

                # Generate a random index
                index = randint(0,len(current_folder_imgs)-1)

                try:
                    # Select an image
                    selected_img = current_folder_imgs[index]

                    # Create source and destination path 
                    src_image_path = os.path.join(currentFolder, selected_img)
                    dst_image_path = os.path.join(destination_folder, selected_img)

                    # Move file from source to destination
                    shutil.move(src_image_path, dst_image_path) 
                    
                    # Increase counter
                    count += 1

                except IndexError:
                    pass  

def oneHotEncoding(path_source:str, img_classes:dict, size:tuple=(128,128,3))->tuple[np.ndarray, np.ndarray]:
    '''
    Transform data into One-hot Encoding
    path_source - source folder path
    img_classes - dictionary with classes and corresponding numbers
    size - size of the images. All images must be in the same size and extension. Default is (128,128,3)
    '''

    # Get image size and number of classes
    image_size = size[0]*size[1]*size[2]
    nr_of_classes = len(img_classes)

    # Create empty arrays for X and Y 
    X_data = np.empty((0, image_size), dtype=np.uint8)
    Y_data = np.empty((0, nr_of_classes), dtype=np.uint8)
    
    for folder in os.listdir(path_source):
        currentFolder = os.path.join(path_source, folder)
        if (os.path.isdir(currentFolder)) and (not '.' in folder):
            
            for item in os.listdir(currentFolder):
                f_name, f_ext = os.path.splitext(item)
                if f_ext == '.png':
    
                    # Open image in RGB mode
                    image_path = os.path.join(currentFolder,item)
                    image = cv2.imread(image_path)
                    image = image.copy()
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                  
                    # SET UP THE DATASET
                    # Vectorize the image and convert it into a list
                    img_arr = np.ndarray.flatten(image)

                    # Get the string of the image expression
                    if '_NA_'in f_name:
                        class_name, expression = f_name.split("_NA_")
                    elif '_WG_' in f_name:
                        class_name, expression = f_name.split("_WG_")
                    else:
                        class_name, expression = f_name.split("_")
                   
                    # Create a zero vector and set the value 1 for the corresponding class
                    label = np.zeros((1,nr_of_classes), dtype=np.uint8)
                    label[0][img_classes[class_name]]=1
                    
                    # Add items to the dataset
                    X_data = np.concatenate((X_data, img_arr.reshape(1,-1)), axis=0)
                    Y_data = np.concatenate((Y_data, label), axis=0)
   
    return X_data, Y_data

def shuffleData(x_data: np.ndarray, y_data: np.ndarray) -> np.array:
    assert x_data.shape[0] == y_data.shape[0]
    data = np.concatenate((x_data, y_data), axis=1)

    # Shuffle data
    np.random.shuffle(data)

    # Split shuffled data 
    x_data = data[:, :x_data.shape[1]]
    y_data = data[:, -(y_data.shape[1]):]

    return x_data, y_data

def save_sets(dataset: Tuple[Tuple[np.ndarray]], preprocess_file: str):

    with open (preprocess_file, "wb") as f:
        pickle.dump(dataset, f)
    