# Neuron Image Detection and Co-localization
## Grinberg Neurosciences Lab at UCSF

repo name: ucsf_gringberg_neuro_images

This project was created to build a tool to identify human flourescense-labeled neurons in microscopy slides. 

Ground truth data for this project was provided by the Grinberg Lab at UCSF and consisted of 200+ human expertly-labeled images for model training and validation.


## Repo Overview
The project is divided into 3 folders for development work. The last folder 'ucsf_production' will host the final code at completion.

Three work folders:
1. early_development: computer vision algorithm development associated with stanford's computer vision lectures
2. ucsf_development : neural net development using ucsf training materials
3. ucsf_production: work pertianing to actual data

Project Phases
1. NeuN-stained neuron segmentation
2. Colocalization of neurons with other markers
3. Package tool for utilization in NIH's open source ImageJ microscopy software.


## Phase 1 - Segmentation
Python
Fast.ai library
U-Net model architecture
![U-Net](U-net.png)
[image source](https://arxiv.org/pdf/1505.04597.pdf)


## Phase 2 - Colocalization




## Phase 3 - ImageJ



