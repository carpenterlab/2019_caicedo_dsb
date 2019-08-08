import sys
import os
import os.path
    
import numpy as np
import pandas as pd
    
import tensorflow as tf
    
import keras.backend
import keras.callbacks
import keras.layers
import keras.models
import keras.optimizers
    
import utils.model_builder
import utils.data_provider
import utils.metrics
import utils.objectives
import utils.dirtools
import utils.evaluation
    
import skimage.io
import skimage.morphology
import skimage.segmentation


def run(config_vars, data_partitions, experiment_name, partition, GPU="2"):

    # Device configuration
    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = GPU
    session = tf.Session(config = configuration)
    
    # apply session
    keras.backend.set_session(session)

    # # Step 02
    # # Training a U-Net model    
    
    train_gen = utils.data_provider.random_sample_generator(
        config_vars["normalized_images_dir"],
        config_vars["boundary_labels_dir"],
        data_partitions["training"],
        config_vars["batch_size"],
        config_vars["pixel_depth"],
        config_vars["crop_size"],
        config_vars["crop_size"],
        config_vars["rescale_labels"]
    )
    
    val_gen = utils.data_provider.single_data_from_images(
         config_vars["normalized_images_dir"],
         config_vars["boundary_labels_dir"],
         data_partitions["validation"],
         config_vars["val_batch_size"],
         config_vars["pixel_depth"],
         config_vars["crop_size"],
         config_vars["crop_size"],
         config_vars["rescale_labels"]
    )
    
    model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
    
    loss = utils.objectives.weighted_crossentropy
    
    metrics = [keras.metrics.categorical_accuracy, 
               utils.metrics.channel_recall(channel=0, name="background_recall"), 
               utils.metrics.channel_precision(channel=0, name="background_precision"),
               utils.metrics.channel_recall(channel=1, name="interior_recall"), 
               utils.metrics.channel_precision(channel=1, name="interior_precision"),
               utils.metrics.channel_recall(channel=2, name="boundary_recall"), 
               utils.metrics.channel_precision(channel=2, name="boundary_precision"),
              ]
    
    optimizer = keras.optimizers.RMSprop(lr=config_vars["learning_rate"])
    
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    
    callback_csv = keras.callbacks.CSVLogger(filename=config_vars["csv_log_file"])
    
    callbacks=[callback_csv]
    
    # TRAIN
    statistics = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=config_vars["steps_per_epoch"],
        epochs=config_vars["epochs"],
        validation_data=val_gen,
        validation_steps=int(len(data_partitions["validation"])/config_vars["val_batch_size"]),
        callbacks=callbacks,
        verbose = 1
    )
    
    model.save_weights(config_vars["model_file"])
    
    print('Training Done! :)')
    
    
    # # Step 03
    # # Predict segmentations
        
    image_names = [f for f in data_partitions[partition] if f.startswith("IXM")]
    image_names = [os.path.join(config_vars["normalized_images_dir"], f) for f in image_names]#data_partitions[partition]]
    
    imagebuffer = skimage.io.imread_collection(image_names)
    
    images = imagebuffer.concatenate()
    
    dim1 = images.shape[1]
    dim2 = images.shape[2]
    
    images = images.reshape((-1, dim1, dim2, 1))
    
    images = images / 255
    
    model = utils.model_builder.get_model_3_class(dim1, dim2)
    model.load_weights(config_vars["model_file"])
    
    predictions = model.predict(images, batch_size=1)
    
    for i in range(len(images)):
    
        filename = imagebuffer.files[i]
        filename = os.path.basename(filename)
        
        probmap = predictions[i].squeeze()
        
        skimage.io.imsave(config_vars["probmap_out_dir"] + filename, probmap)
        
        pred = utils.metrics.probmap_to_pred(probmap, config_vars["boundary_boost_factor"])
    
        label = utils.metrics.pred_to_label(pred, config_vars["cell_min_size"])
        
        skimage.io.imsave(config_vars["labels_out_dir"] + filename, label)
    
    
    # # Step 04
    # # Evaluation of performance
    
    all_images = data_partitions[partition]
    #all_images = [f for f in data_partitions[partition] if f.startswith("IXM")]
    
    
    #results = pd.DataFrame(columns=["Image", "Threshold", "Precision"])
    #false_negatives = pd.DataFrame(columns=["False_Negative", "Area"])
    #splits_merges = pd.DataFrame(columns=["Image_Name", "Merges","Splits"])
    results = pd.DataFrame(columns=["Image", "Threshold", "F1", "Jaccard", "TP", "FP", "FN"])
    false_negatives = pd.DataFrame(columns=["False_Negative", "Area"])
    splits_merges = pd.DataFrame(columns=["Image_Name", "Merges", "Splits"])
    
    for image_name in all_images:
        img_filename = os.path.join(config_vars["raw_annotations_dir"], image_name)
        ground_truth = skimage.io.imread(img_filename)
        if len(ground_truth.shape) == 3:
            ground_truth = ground_truth[:,:,0]
        
        ground_truth = skimage.morphology.label(ground_truth)
        
        pred_filename = os.path.join(config_vars["labels_out_dir"], image_name)
        prediction = skimage.io.imread(pred_filename) #.replace(".png",".tiff"))
        
        if config_vars["object_dilation"] > 0:
            struct = skimage.morphology.square(config_vars["object_dilation"])
            prediction = skimage.morphology.dilation(prediction, struct)
        elif config_vars["object_dilation"] < 0:
            struct = skimage.morphology.square(-config_vars["object_dilation"])
            prediction = skimage.morphology.erosion(prediction, struct)
            
        ground_truth = skimage.segmentation.relabel_sequential(ground_truth[30:-30,30:-30])[0] # )[0] #
        prediction = skimage.segmentation.relabel_sequential(prediction[30:-30,30:-30])[0] # )[0] #
        
        results = utils.evaluation.compute_af1_results(
            ground_truth, 
            prediction, 
            results, 
            image_name
        )
        
        false_negatives = utils.evaluation.get_false_negatives(
            ground_truth, 
            prediction, 
            false_negatives, 
            image_name
        )
        
        splits_merges = utils.evaluation.get_splits_and_merges(
            ground_truth, 
            prediction, 
            splits_merges, 
            image_name
        )
        
    
    # # Report of results
    
    output = {}

    results = results[results["Threshold"] < 0.95]
    average_performance = results.groupby("Threshold").mean().reset_index()
    output["Average_F1"] = average_performance["F1"].mean()
    output["Jaccard"] = average_performance["Jaccard"].mean()

    false_negatives = false_negatives[false_negatives["False_Negative"] == 1]
    
    missed = false_negatives.groupby(
        pd.cut(
            false_negatives["Area"], 
            [0,250,625,900,10000], # Area intervals
            labels=["Tiny nuclei","Small nuclei","Normal nuclei","Large nuclei"],
        )
    )["False_Negative"].sum()
    
    output["Missed"] = missed
    output["Splits"] = np.sum(splits_merges["Splits"])
    output["Merges"] = np.sum(splits_merges["Merges"])

    return output
    
