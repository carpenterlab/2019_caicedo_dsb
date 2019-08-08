
# coding: utf-8

# # Step 03
# # Predict segmentations

import sys
import os
import os.path

import numpy as np
import pandas as pd

import skimage.io
import skimage.exposure
import skimage.morphology
import skimage.measure
import skimage.transform
import skimage.segmentation

import tensorflow as tf
import keras

import utils.metrics
import utils.model_builder

# # Configuration
PI = 3.1415926539
HALF_SIDE = 96 # pixels
SAVE_OUTPUT = "locations"
IMG_EXT = ".tif"

from config import config_vars

if len(sys.argv) < 5:
    print("Use: python batch_prediction.py experiment_name image_list.csv input_dir output_dir")
    sys.exit()
else:
    experiment_name = sys.argv[1]
    image_list = pd.read_csv(sys.argv[2])
    input_dir = sys.argv[3]
    output_dir = sys.argv[4]

# Partition of the data to make predictions (test or validation)

config_vars = utils.dirtools.setup_experiment(config_vars, experiment_name)

# Configuration to run on GPU
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "0"

session = tf.Session(config = configuration)

# apply session
keras.backend.set_session(session)


# Load model
image = skimage.io.imread(input_dir + image_list.loc[0,"DNA"])
dim1 = image.shape[0]
dim2 = image.shape[1]

# build model and load weights
model = utils.model_builder.get_model_3_class(dim1, dim2)
model.load_weights(config_vars["model_file"])
model.summary()

# 

# # Load images and run predictions

total_num_images = len(image_list)
# The current logic of the code breaks if batch_size > 1
batch_size = 1


# Check that images have not been processed before
print("Verifying image list")
image_list["Done"] = False
for k,r in image_list.iterrows():
    img_name = image_list.loc[k,"DNA"]
    outfile = output_dir + img_name.replace(IMG_EXT, ".csv")
    if os.path.isfile(outfile):
        image_list.loc[k,"Done"] = True

print("Total images:",image_list.shape[0])
image_list = image_list[~image_list["Done"]]
print("Pending processing:", image_list.shape[0])


i = 0
while i < total_num_images:
    batch = image_list.loc[i:i+batch_size-1, "DNA"].values
    if i + batch_size < total_num_images:
        i += batch_size
    else:
        batch_size = total_num_images - i
        i += batch_size

    image_names = [input_dir + b for b in batch]

    # Check that images exist
    missing = [k for k in range(len(image_names)) if not os.path.isfile(image_names[k])]

    # Filter images missing
    good_to_go = []
    for j in range(len(batch)):
        if j in missing:
            print("Image missing:", batch[j])
        else:
            good_to_go.append(j)

    image_names = [image_names[k] for k in good_to_go]
    batch = [batch[k] for k in good_to_go]

    if len(batch) == 0:
        continue

    # Load images
    imagebuffer = skimage.io.imread_collection(image_names)
    images = imagebuffer.concatenate()
    images = images.reshape((-1, dim1, dim2, 1))

    # Normalize pixels
    images = images / np.max(np.max(np.max(images, axis=-1), axis=-1), axis=-1)

    # Normal prediction time
    predictions = model.predict(images, batch_size=len(batch))

    # # Transform predictions to label matrices

    for j in range(len(batch)):
        # Determine whether the image has been processed before
        filename = output_dir + batch[j].replace(IMG_EXT,".csv")
        if os.path.isfile(filename):
            print("Image", batch[j], "already done")
            continue

        print("Image",batch[j])

        os.makedirs("/".join(filename.split("/")[0:-1]), exist_ok=True)
        probmap = predictions[j].squeeze()
        pred = utils.metrics.probmap_to_pred(probmap, config_vars["boundary_boost_factor"])
        label = utils.metrics.pred_to_label(pred, config_vars["cell_min_size"])

        # Apply object dilation
        if config_vars["object_dilation"] > 0:
            struct = skimage.morphology.square(config_vars["object_dilation"])
            label = skimage.morphology.dilation(label, struct)
        elif config_vars["object_dilation"] < 0:
            struct = skimage.morphology.square(-config_vars["object_dilation"])
            label = skimage.morphology.erosion(label, struct)
    
        label = label.astype(np.int16)
        if SAVE_OUTPUT == "masks" or SAVE_OUTPUT == "both":
            skimage.io.imsave(filename, label)

        # Save object properties
        if SAVE_OUTPUT == "locations" or SAVE_OUTPUT == "both":
            os.makedirs("/".join(filename.split("/")[0:-1]), exist_ok=True)
            nuclei_df = pd.DataFrame(columns=["Nuclei_Location_Center_X","Nuclei_Location_Center_Y","Orientation"])
            regions = skimage.measure.regionprops(label)

            idx = 0
            for region in regions:
                row = int(region.centroid[0])
                col = int(region.centroid[1])
                if row - HALF_SIDE > 0 and row + HALF_SIDE < dim1 and col - HALF_SIDE > 0 and col + HALF_SIDE < dim2:
                    angle = int(90 - (180 * region.orientation) / PI)
                    nuclei_df.loc[idx] = {"Nuclei_Location_Center_X": col, "Nuclei_Location_Center_Y": row, "Orientation": angle}
                    idx += 1

            nuclei_df.to_csv(filename, index=False)


