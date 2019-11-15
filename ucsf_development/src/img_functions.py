import numpy as np
from fastai.vision import open_mask, open_image
import math

# Display images
from IPython.display import Image, display
import PIL.Image


# function to show that training images were loaded properly
def train_img_test(img_file_list, open_image):
    '''
    print info on the image files being loaded into the databunch
    dependency:  from fastai.vision import *
    input: 
        img_file_list   list of PosixPath files of images
        label           boolean. Set to true if label images
    '''
    # print 3 files but if < 3 then print that many
    stop = _file_aux(img_file_list)
    
    # image file info
    for i in range(stop):
        # file name
        print("{}:\tName: {}".format(i, img_file_list[i]))
        
        # converts to type = fastai.vision.image.Image
        img = open_image(img_file_list[i])  
        
        # sow image size/shape
        print("\tShape: {}".format(img.shape))
        
        # show image
        img.show(figsize=(5, 5))
        print("")
    # end function


def _file_aux(img_file_list):
    
    # print 3 files but if < 3 then print that many
    n_img = len(img_file_list)
    print("There are {} image files.".format(n_img))
    if n_img > 3:
        stop = 3
    else:
        stop = n_img
    warning = None
    print("First {} image files:".format(stop))
    return stop



def label_info(img_names, lbl_names, get_y_fn=None):
    # Display an example mask using fastai function
    # open_mask is used for segmentation labels because label data are integers, not floats
    n_img = len(img_names)
    if n_img > 3:
        stop = 3
    else:
        stop = n_img
    print("There are {} images in the folder.".format(n_img))
    print("First {} image files:".format(stop))
    color_classes = set()
    for i in range(stop):
        train = open_image(img_names[i])
        mask = open_mask(get_y_fn(img_names[i]))   
        c_classes = np.unique(mask.data)
        color_classes.update(c_classes)
        print("---------------------")
        print(i)
        print("Name: {}".format(img_names[i]))    # training image file name
        print("Size: Train - {} \tMask - {}".format(np.array(train.shape[1:3]), 
                                                    np.array(mask.shape[1:3])))
        print("Unique Color Classes:{}".format(c_classes))
        print(mask.data)
        mask.show(figsize=(5, 5), alpha=1)
    print("---------------------")
    print("Total number of segementation classes: {}".format(len(color_classes)))
    print("Classes: {}".format(color_classes))
    print("---------------------")
    print("Note: default color for display is blue in fastai.")



# convert rgb image to greyscale
# --- not complete, don't use ---
def image_to_single_channel(path_folder, new_dir_name):
    '''
    converts all image files in a directory from rgb to single channel (greyscale) using PIL library
    input:
        path_folder:   path to folder holding rgb files to be converted
        new_dir_name:  name of new directoryc
    '''
    img = pil_image.open('image.png').convert('LA')
    img.save('greyscale.png')
    
    
    
    


def display_img(folder_path, file_names, width=100, height=100):
    '''
    displays images in line
    ---
    folder_path:   str. folder that contains the images. ex: /Users/shared/.fastai/data/camvid_tiny/labels
    file_names:    list. filenames as strings.
    '''
    for imageName in file_names:
        imageName = folder_path+'/'+imageName
        display(Image(filename=imageName, width=width, height=height))
        

def img_window(img_name):
    '''
    # Display image in a separate window using PIL library
    depend:  from PIL import Image as pil_image
    input:  img_name   posixPath file name
    '''
    im = PIL.Image.open(img_name)
    im.show()


def img_slicer(save_dir, path_img):
    '''docs here
    depend: import math
    '''
    for img_file in get_image_files(path_img):
        img = pil_image.open(img_file)
        width_interval = math.floor(img.width/300)
        height_interval = math.floor(img.height/300)
        for width in range(width_interval):
            for height in range(height_interval):
                box = width*300,height*300,(width+1)*300,(height+1)*300
                region = img.crop(box)
                # 300-300-TrainingData_1_original.tif
                fname = "{}-{}-{}".format(width*300, height*300,img_file.parts[-1]) # change to dynamic
                region.save(save_dir+fname)

    
def save_img_greyscale(fpath, fname, save_dir):
    # converts image to grey scale and saves to a folder
    # only to be run once
    # input: fpath, fname, save_dir  ->  all strings
    img = pil_image.open(fpath+'/'+fname).convert('L') # 'L' = 8-bit pixels, black and white
    print(img.size)
    print(img.mode)
    img.save(save_dir+'/'+fname)

'''
# image display functions in PIL
# ---
pil_image.open(fname)  # path and filename...verify
img = PIL.Image.open(fname)  # single file - assigned to PIL image object (PIL.Image.Image), can be manipulated with other PIL functions

'''