import numpy as np
from fastai.vision import open_mask, open_image, get_image_files
import math

# Display images
from IPython.display import Image, display
from PIL import Image as pil_image

from pathlib import Path


# function to show that training images were loaded properly
def train_img_test(img_file_names, open_image, n_display=3):
    '''
    print info on the image files being loaded into the databunch
    dependency:  from fastai.vision import open_mask, open_image, get_image_files
    input: 
        img_file_names   list of PosixPath files of images
        label           boolean. Set to true if label images
    '''
    # print info in n files but if < n_display then print that many
    stop = _file_aux(img_file_names, n_display)
    
    image_sizes = []
    # image meta data
    for i in range(len(img_file_names)):
        
        # converts to type = fastai.vision.image.Image
        img = open_image(img_file_names[i])
        
        # get image sizes
        img_width, img_height = img.shape[1], img.shape[2]
        if i==0:
            width_min = img_width
            height_min = img_height
        # set min size values
        if img_width < width_min:
            width_min = img_width
        if img_height < height_min:
            height_min = img_height
        
        image_sizes.append((img_width, img_height))
        
        # display single file meta data
        if i <= stop:
            # print file name
            print("{}:\tName: {}".format(i, img_file_names[i]))
        
            # show image size/shape
            print("\tShape: {}".format(img.shape))
        
            # show image
            img.show(figsize=(5, 5))
            print("")
    
    print("Minimum width: {}".format(width_min))
    print("Minimum height: {}".format(height_min))
    print("Size of images in this directory: {}".format(image_sizes))
        
    # end function


def _file_aux(img_file_names, n_display):
    ''' works with train_img_test function'''
    # print 3 files but if < 3 then print that many
    n_img = len(img_file_names)
    print("There are {} image files.".format(n_img))
    if n_img > n_display:
        stop = n_display
    else:
        stop = n_img
    warning = None
    print("First {} image files:".format(stop))
    return stop



def label_info(img_file_names, lbl_file_names, get_y_fn=None, n_display=3):
    ''' Display an example mask using fastai function '''
    
    stop = _file_aux(img_file_names, n_display)
    
    # open_mask is used for segmentation labels because label data are integers, not floats
    '''n_img = len(img_names)
    if n_img > display_n:
        stop = display_n
    else:
        stop = n_img
    print("There are {} images in the folder.".format(n_img))
    print("First {} image files:".format(stop))'''
    
    color_classes = set()
    for i in range(stop):
        train = open_image(img_file_names[i])   # open_image is fastai method
        mask = open_mask(get_y_fn(img_file_names[i]))   
        c_classes = np.unique(mask.data)
        color_classes.update(c_classes)
        print("---------------------")
        print(i)
        print("Name: {}".format(img_file_names[i]))    # training image file name
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
    displays images inline
    ---
    folder_path:   str. folder that contains the images. ex: /Users/shared/.fastai/data/camvid_tiny/labels
    file_names:    list. filenames as strings.
    '''
    for imageName in file_names:
        imageName = folder_path+'/'+imageName
        display(Image(filename=imageName, width=width, height=height))
        

def img_window(img_name):
    '''
    Display image in a separate window using PIL library.
    depend:  from PIL import Image as pil_image
    input:  img_name   posixPath file name
    '''
    im = PIL.Image.open(img_name)
    im.show()

    
def save_img_greyscale(fpath, fname, save_dir):
    ''' converts image to grey scale and saves to a folder. '''
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

# 

def img_slicer(size, save_dir, path_img):
    '''
    The following approach breaks an large image into equal sizexsize boxes, but throws 
    out the "leftover" image space.
    - opens file, crops iteratively, saves 
    depends:
        from PIL import Image
        from pathlib import Path
        from fastai.vision import get_image_files
        import math
    input:
        size:      int. square size of cropped image output
        save_dir:  str. location to save cropped images.
        path_img:  posixpath object. location of dir of image files to be cropped.
    ''' 
    # open each image in the file path
    for img_file in get_image_files(path_img):
        # check for dir and make is needed
        p=Path(save_dir)
        if p.exists()==False:
            p.mkdir()
        img = pil_image.open(img_file)
        # dertermine number of iterations width and height wise to proceed
        width_interval = math.floor(img.width/size)
        height_interval = math.floor(img.height/size)
        # 
        for width in range(width_interval):
            for height in range(height_interval):
                # set corner location of square box to be cropped
                box = width*size,height*size,(width+1)*size,(height+1)*size
                # crop this region only
                region = img.crop(box)
                # give cropped region a filename and save to appointed directory
                fname = "{}-{}-{}".format(width*size, height*size,img_file.parts[-1]) # change to dynamic
                region.save(save_dir+fname)

