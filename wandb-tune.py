import gin
import wandb
import pathlib
from utils import utils_params
from train import train


# os.environ['WANDB_MODE'] = 'offline'

# Name of the folder and wandb project for hyperparameter tuning.
dir_name = "run"


def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f"{key}={value}")

        # generate folder structures
        run_paths = utils_params.gen_run_folder(dir_name)

        # gin-config
        gin_config_path = pathlib.Path(__file__).parent / "configs" / "config.gin"
        gin.parse_config_files_and_bindings([gin_config_path], bindings)
        utils_params.save_config(run_paths["file_gin"], gin.config_str())

        # start training
        train(run_paths, dir_name=dir_name, train_flag=True)
        utils_params.move_logs(run_paths)

# Configuration for the hyperparameter tuning.
sweep_config = {
    "program": "wandb-tune.py",
    "command": ["python3", "wandb-tune.py"],
    "name": "human_activity",
    "parameters": {
        "HAPTDataModule.WIN_SHIFT": {"values": [100, 125, 128, 150]},
        "HAPTDataModule.WIN_SIZE": {"values": [200, 250, 256, 300]},
        "get_classifier.hidden_size": {"values": [2 * i for i in range(3, 40)]},
        "train.EPOCHS": {
            "distribution": "int_uniform",
            "min": 10,
            "max": 60,
        },
        "HAPTDataModule.BATCH_SIZE": {"values": [16, 32, 64, 128]},
        "get_classifier.num_layers": {"values": [2, 3, 4]},
        "get_classifier.LEARNING_RATE": {
            "distribution": "log_uniform",
            "min": -9.21,
            "max": -2.3,
        },
        "get_classifier.DROPOUT_RATE": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.4,
        },
        "get_classifier.REGULARIZATION": {
            "distribution": "log_uniform",
            "min": -9.21,
            "max": -2.3,
        },
    },
    "metric": {"goal": "maximize", "name": "validation_accuracy"},
    "method": "bayes",
}

# Create a sweep 
sweep_id = wandb.sweep(sweep_config, project=dir_name)

# Perfrom the sweep
wandb.agent(sweep_id, function=train_func, count=50)
