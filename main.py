import gin
import absl
from absl import app
import pathlib
from utils import utils_params
import wandb
from train import train
import os



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_boolean(
    name="train", default=False, help="Specify whether to train a model."
)
absl.flags.DEFINE_boolean(
    name="eval", default=False, help="Specify whether to evaluate a model."
)
absl.flags.DEFINE_string(
    name="dir_name", default="har_run", help="Specify the name of the run folder."
)


def main(argv):
    """
    The main function. It creates the run paths, sets the config and calls the train function.
    """

    # prerequisites
    run_paths = utils_params.gen_run_folder(FLAGS.dir_name)
    utils_params.save_config(run_paths["file_gin"], gin.config_str())

    wandb.init(project=FLAGS.dir_name)

    # Train and test
    train(
        run_paths, dir_name=FLAGS.dir_name, train_flag=FLAGS.train, eval_flag=FLAGS.eval
    )
    utils_params.move_logs(run_paths)


if __name__ == "__main__":
    gin_config_path = pathlib.Path(__file__).parent / "configs" / "config.gin"
    gin.parse_config_files_and_bindings([gin_config_path], [])
    app.run(main)
