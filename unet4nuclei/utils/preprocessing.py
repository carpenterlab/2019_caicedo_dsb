import os

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


