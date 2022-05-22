"""
Keep track of utils
"""
import os
import shutil
import datetime


def gen_run_folder(path_model_id=datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")):
    """Create directory for the run, log files and checkpoints.

    Returns:
        (dict): Dictionary containing names of directories and their paths, paths are pathlib.Path.
    """

    run_paths = {}
    path_model_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir,"experiments")
    )
    if not os.path.exists(path_model_root):
        os.makedirs(path_model_root, exist_ok=True)

    run_paths["root"] = path_model_root

    # directories
    run_paths["path_model_id"] = os.path.join(path_model_root, path_model_id)

    run_paths["path_datasets"] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "datasets")
    )
    run_paths["path_HAPT"] = os.path.join(run_paths["path_datasets"], "HAPT")

    run_paths["path_temp"] = os.path.join(run_paths["path_model_id"], "temp")
    run_paths["path_logs"] = os.path.join(run_paths["path_temp"], "logs")
    run_paths["path_gin"] = os.path.join(run_paths["path_temp"], "gins")

    # files
    run_paths["file_run_log"] = os.path.join(run_paths["path_logs"], "run.log")
    run_paths["file_training_log"] = os.path.join(
        run_paths["path_logs"], "training.log"
    )
    run_paths["file_evaluation_log"] = os.path.join(
        run_paths["path_logs"], "evaluation.log"
    )
    run_paths["file_gin"] = os.path.join(run_paths["path_gin"], "config_operative.gin")

    # Create folders
    for name, path in run_paths.items():
        if ("path_" in name) and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    # Create files
    for name, path in run_paths.items():
        if "file_" in name:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8"):
                pass  # atm file creation is sufficient

    return run_paths


def move_logs(run_paths):

    # get latest run version of the model
    versions = os.listdir(run_paths["path_model_id"])
    versions_filepath = []
    for v in versions:
        folder = os.path.join(run_paths["path_model_id"], v)
        versions_filepath.append(folder)
    latest_version = max(versions_filepath, key=os.path.getmtime)

    # directories inside the latest version of the model
    run_paths["path_version"] = latest_version
    run_paths["path_v_logs"] = os.path.join(run_paths["path_version"], "logs")
    run_paths["path_v_gin"] = os.path.join(run_paths["path_version"], "gins")

    # create folders
    for name, path in run_paths.items():
        if ("path_" in name) and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    shutil.move(
        run_paths["file_run_log"], os.path.join(run_paths["path_v_logs"], "run.log")
    )
    shutil.move(
        run_paths["file_training_log"],
        os.path.join(run_paths["path_v_logs"], "training.log"),
    )
    shutil.move(
        run_paths["file_evaluation_log"],
        os.path.join(run_paths["path_v_logs"], "evaluation.log"),
    )
    shutil.move(
        run_paths["file_gin"],
        os.path.join(run_paths["path_v_gin"], "config_operative.gin"),
    )


def save_config(path_gin, config):
    """
    function to write and save config file
    """
    with open(path_gin, "w", encoding="utf-8") as f_config:
        f_config.write(config)
