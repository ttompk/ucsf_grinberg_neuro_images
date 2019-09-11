README.md

# USCF Grinberg Lab Development

This directory uses data from the Grinberg lab.

## Manually labeling cell bodies.

Use GIMP to create a map for identifying red, green, and multi colored cells. The GIMP process is generic and can be applied to any number of classes. Repeat the GIMP process if more than one mask layer.

### GIMP Process

Within  GIMP:
1. Load the black and white cell body mask pic.
2. Invert the colors. From 'Colors' -> 'Invert'
3. Add transparency layer. From 'Layer' -> 'Transparency' -> 'Add Alpha Channel'
4. Select all the cell bodies. Using the 'select by color tool' tool from the Toolbox, click on a large cell body and drag very slightly before releasing. After a moment all the cell bodies will have a moving boder around each cell body.
5. Remove the selected cell bodies from the image. From 'Edit' -> 'Clear'.
6. Open the corresponding color image.
7. Click back on the black and white picture. Select the entire image. From 'Select' -> 'None', then 'Select' -> 'All'. Copy the image.
8. Paste the black and white mask over the color image. Click on the color image. 'Edit' -> 'Paste As' -> 'New Layer'. This may take some time to process. Once complete, only the cell bodies identified in the mask will be shown. 
9. Export the new image as a png. Remember to change the name! From 'File' -> 'Export'. Select the proper directory, remove the extension from the filename, and add the following suffix: '_train.png'.
10. An 'Export Image as png' dialog box will appear. Select 'Export'.

Now the images should be ready for class labeling.

### Label the classes with boxes 

The 'draw_boxes.py' is designed to allow the user to draw boxes around the cell bodies in the new'_train.png' file. 

Run 'draw_boxes.py' from the terminal. Follow the instructions and prompts when asked. Because this project has three classes of cell bodies, you will need to run the program three times...at least until I change the code to ask the number of classes.  

All the images in the folder are available for annotating. To move to the next box press the right arrow key '->'. 
- Note: everytime you draw a box, the box location is saved, however once you move to the next image, you cannot go back to the previous image and see where you have already drawn boxes!!

After you are satisfied with your labeling, press the 's' key. This will write a .csv file to the current working directory. 

Repeat above for each class.

### Neural Net

The 'image_processing.ipynb' file allows the user to verify the boxes are where we want them to be.  
The notebook combines the 3 classes together into one dataframe and plots them on top of the image of interest. 

