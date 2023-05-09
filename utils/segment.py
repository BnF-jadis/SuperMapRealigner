# 2020, 2023, Jadis Project
# Coded by Remi Petitpierre https://github.com/RPetitpierre
# For Bibliothèque nationale de France (BnF) and EPFL, Swiss Federal Institute of Technology in Lausanne

import cv2, os
import numpy as np
from skimage.morphology import skeletonize

def makeImagePatches(image: np.ndarray, patches_path: str = '', export: bool = True,
                     patch_size: int = 1000, border_width: int = 200):
    ''' Reconstitution of the map based on the prediction on image patches.
    Input(s):
        image: original full-size map
        patches_path: path to the folder where the image patches will be saved
        export: if True, the patches are saved to patches_path
    Output(s):
        rows: number of rows of patches
        cols: number of columns of patches
        patches: if they are not exported, the image patches are returned
    '''
    
    core_size = patch_size - border_width
    
    rows = 1 + ((image.shape[0]-border_width+1)//core_size)
    cols = 1 + ((image.shape[1]-border_width+1)//core_size)
    
    patches = []
    i = 0
    for row in range(rows):
        for col in range(cols):
            patch = image[row*core_size:patch_size+row*core_size, 
                          col*core_size:patch_size+col*core_size]
            if patch.shape[:2] != (patch_size, patch_size):
                background = np.zeros((patch_size, patch_size, image.shape[2]))
                background[0:patch.shape[0], 0:patch.shape[1]] = patch
                patch = background
            
            if export:
                cv2.imwrite(os.path.join(patches_path, str(row) + '_' + str(col) + '.tif'), patch.astype('uint8'))
                i += 1
            else:
                patches.append(patch.astype('uint8'))
    if export:        
        return rows, cols
    else:
        return rows, cols, patches

def preprocess_segmented_maps(image_path: str, preproc_folder: tuple, max_size: int = 1400):
    """
    Preprocess segmented maps by resizing and skeletonizing the input image.
    
    Input(s):
        image_path (str): Path to the input image file.
        max_size (int, optional): Max image size factor
    Output(s):
        tuple: (image_name, image_scale, image_shape)
    """
    
    # Read image and get its properties
    image = cv2.imread(image_path)
    image_shape = image.shape 
    image_name = os.path.basename(image_path)
    image_scale = max(image.shape)/max_size
    
    # Resize and skeletonize the image
    resized_image = cv2.resize(image, (0, 0), fx=1/image_scale, fy=1/image_scale)
    skeletonized_image = skeletonize(resized_image)
        
    # Save the skeletonized image
    cv2.imwrite(os.path.join(*preproc_folder, image_name), skeletonized_image)
    
    return image_name, image_scale, image_shape