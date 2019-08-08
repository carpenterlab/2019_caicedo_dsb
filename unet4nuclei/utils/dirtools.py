import os
import glob
import random 

def create_image_lists(dir_raw_images, fraction_train = 0.5, fraction_validation = 0.25):
    file_list = os.listdir(dir_raw_images)

    if (fraction_train + fraction_validation >= 1):
        print("fraction_train + fraction_validation is > 1!")
        print("setting fraction_train = 0.5, fraction_validation = 0.25")
        fraction_train = 0.5
        fraction_validation = 0.25
        
    fraction_test = 1 - fraction_train - fraction_validation

    image_list = [x for x in file_list if x.endswith("png") ]

    random.shuffle(image_list)

    index_train_end = int( len(image_list) * fraction_train)
    index_validation_end = index_train_end + int(len(image_list) * fraction_validation)

    # split into two parts for training and testing 
    image_list_train = image_list[0:index_train_end]
    image_list_test = image_list[index_train_end:(index_validation_end)]
    image_list_validation = image_list[index_validation_end:]
    return(image_list_train, image_list_test, image_list_validation)


def write_path_files(file_path, list):
    with open(file_path, 'w') as myfile:
        for line in  list: myfile.write(line + '\n')


def setup_working_directories(config_vars):

    ## Expected raw data directories:
    config_vars["raw_images_dir"] = os.path.join(config_vars["root_directory"], 'raw_images/')
    config_vars["raw_annotations_dir"] = os.path.join(config_vars["root_directory"], 'raw_annotations/')

    ## Split files
    config_vars["path_files_training"] = os.path.join(config_vars["root_directory"], 'training.txt')
    config_vars["path_files_validation"] = os.path.join(config_vars["root_directory"], 'validation.txt')
    config_vars["path_files_test"] = os.path.join(config_vars["root_directory"], 'test.txt')

    ## Transformed data directories:
    config_vars["normalized_images_dir"] = os.path.join(config_vars["root_directory"], 'norm_images/')
    config_vars["boundary_labels_dir"] = os.path.join(config_vars["root_directory"], 'boundary_labels/')

    return config_vars


def read_data_partitions(config_vars, load_augmented=True):
    with open(config_vars["path_files_training"]) as f:
        training_files = f.read().splitlines()
        if config_vars["max_training_images"] > 0:
            random.shuffle(training_files)
            training_files = training_files[0:config_vars["max_training_images"]]
        
    with open(config_vars["path_files_validation"]) as f:
        validation_files = f.read().splitlines()
    
    with open(config_vars["path_files_test"]) as f:
        test_files = f.read().splitlines()

    # Add augmented images to the training list
    if load_augmented:
        files = glob.glob(config_vars["root_directory"] + "norm_images/*_aug_*.png")
        files = [f.split("/")[-1] for f in files]
        if config_vars["max_training_images"] > 0:
            augmented = []
            for trf in training_files:
                augmented += [f for f in files if f.startswith(trf.split(".")[0])]
            training_files += augmented
        else:
            training_files += files

    partitions = {
        "training": training_files,
        "validation": validation_files,
        "test": test_files
    }

    return partitions

def setup_experiment(config_vars, tag):

    # Output dirs
    config_vars["experiment_dir"] = os.path.join(config_vars["root_directory"], "experiments/" + tag + "/out/")
    config_vars["probmap_out_dir"] = os.path.join(config_vars["experiment_dir"], "prob/")
    config_vars["labels_out_dir"] = os.path.join(config_vars["experiment_dir"], "segm/")

    # Files
    config_vars["model_file"] = config_vars["root_directory"] + "experiments/" + tag + "/model.hdf5"
    config_vars["csv_log_file"] = config_vars["root_directory"] + "experiments/" + tag + "/log.csv"

    # Make output directories
    os.makedirs(config_vars["experiment_dir"], exist_ok=True)
    os.makedirs(config_vars["probmap_out_dir"], exist_ok=True)
    os.makedirs(config_vars["labels_out_dir"], exist_ok=True)

    return config_vars

