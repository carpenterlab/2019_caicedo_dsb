import os
import utils.dirtools

config_vars = {}

# ************ 01 ************ #
# ****** PREPROCESSING ******* #
# **************************** #

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 01.01 INPUT DIRECTORIES AND FILES

config_vars["root_directory"] = '/data1/image-segmentation/dsb_eval/'

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 01.02 DATA PARTITION INFO

## Maximum number of training images (use 0 for all)
config_vars["max_training_images"] = 0

## Generate partitions?
## If False, load predefined partitions (training.txt, validation.txt and test.txt)
config_vars["create_split_files"] = False

## Randomly choose training and validation images.
## The remaining fraction is reserved for test images.
config_vars["training_fraction"] = 0.5
config_vars["validation_fraction"] = 0.25

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 01.03 IMAGE STORAGE OPTIONS

## Transform gray scale TIF images to PNG
config_vars["transform_images_to_PNG"] = True
config_vars["pixel_depth"] = 8

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 01.04 PRE-PROCESSING OF ANNOTATIONS

## Area of minimun object in pixels
config_vars["min_nucleus_size"] = 5

## Pixels of the boundary (min 2 pixels)
config_vars["boundary_size"] = 2

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 01.05 DATA AUGMENTATION USING ELASTIC DEFORMATIONS

## Elastic deformation takes a lot of times to compute. 
## It is computed only once in the preprocessing. 
config_vars["augment_images"] =  False

## Augmentation parameters. 
## Calibrate parameters using the 00-elastic-deformation.ipynb
config_vars["elastic_points"] = 16
config_vars["elastic_distortion"] = 5

## Number of augmented images
config_vars["elastic_augmentations"] = 10


# ************ 02 ************ #
# ********* TRAINING ********* #
# **************************** #

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 02.01 OPTIMIZATION

config_vars["learning_rate"] = 1e-4

config_vars["epochs"] = 50

config_vars["steps_per_epoch"] = 200

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 02.02 BATCHES

config_vars["batch_size"] = 10

config_vars["val_batch_size"] = 10

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 02.03 DATA NORMALIZATION

config_vars["rescale_labels"] = True

config_vars["crop_size"] = 256

# ************ 03 ************ #
# ******** PREDICTION ******** #
# **************************** #

config_vars["cell_min_size"] = 16

config_vars["boundary_boost_factor"] = 1

# ************ 04 ************ #
# ******** EVALUATION ******** #
# **************************** #

config_vars["object_dilation"] = 3

# **************************** #
# ******** FINAL SETUP ******* #
# **************************** #

config_vars = utils.dirtools.setup_working_directories(config_vars)

