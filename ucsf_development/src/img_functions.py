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
