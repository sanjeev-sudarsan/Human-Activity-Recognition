import gin
from pathlib import PurePath
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from input_pipeline.data_loader import HAPTDataModule
from architecture.models import (
    maxpool_bidirectional_lstm,
    maxpool_lstm,
    simple_lstm,
    bidirectional_lstm,
)

@gin.configurable
def get_classifier(
    model,
    win_size,
    input_size,
    hidden_size,
    num_layers,
    num_classes,
    LEARNING_RATE,
    DROPOUT_RATE,
    REGULARIZATION,
):
    """
    Creates an object of the specified model.

    Parameters:
        model (str): The name of the model.
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden units of the lstm.
        num_layers (int): The number of recurrent layers.
        num_classes (int): The number of classes.
        win_size (int): The size of the sliding window.
        LEARNING_RATE (float): The learning rate of the model.
        DROPOUT_RATE (float): The dropout rate of the model.
        REGULARIZATION (float): The regularization factor of the model.

    Returns:
        classifier (Classifier class): The object of the specified model.
    """

    if model == "simple_lstm":
        classifier = simple_lstm(
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            LEARNING_RATE,
            DROPOUT_RATE,
            REGULARIZATION,
        )
    elif model == "bidirectional_lstm":
        classifier = bidirectional_lstm(
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            LEARNING_RATE,
            DROPOUT_RATE,
            REGULARIZATION,
        )
    elif model == "maxpool_lstm":
        classifier = maxpool_lstm(
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            win_size,
            LEARNING_RATE,
            DROPOUT_RATE,
            REGULARIZATION,
        )
    elif model == "maxpool_bidirectional_lstm":
        classifier = maxpool_bidirectional_lstm(
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            win_size,
            LEARNING_RATE,
            DROPOUT_RATE,
            REGULARIZATION,
        )
    else:
        raise ValueError("Invalid model")
    return classifier


@gin.configurable
def train(
    run_paths,
    train_flag,
    dir_name,
    model,
    resume_model_path,
    evaluation_checkpoint,
    EPOCHS,
    eval_flag=False,
):
    """
    Trains and evaluates the specified model
    Parameters:
        run_paths (dict): Dictionary containing all the required paths.
        train_flag (boolean): If true the specified model is trained.
        dir_name (str): Name of the folder and wandb project for the current run.
        model (str): The name of the model.
        resume_model_path (str): The path of the checkpoint which is used to resume training. If no path is given, it trains from scratch.
        evaluation_checkpoint (str): The path of the checkpoint which is used to resume training.
                                    If no path is given, it evaluates using the current weights of the model.
                                    If no path is given and training has not been performed, it shows an error.
        epochs (int): The number of training epochs.
        eval_flag (boolean): If true the specified model is evaluated. The default value is False.
    """

    logger = TensorBoardLogger(run_paths["root"], version=0, name=dir_name, sub_dir="summaries")
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_accuracy", verbose=True, save_top_k=4, mode="max"
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_accuracy", patience=3, verbose=True, mode="max"
    )
    dataloader = HAPTDataModule(run_paths)

    # Get the trainer
    trainer = Trainer(
        logger=logger,
        gpus=1,
        max_epochs=EPOCHS,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=25,
    )

    # Get the model
    classifier = get_classifier(model, dataloader.win_size)

    if train_flag:
        trainer.fit(classifier, dataloader, ckpt_path=resume_model_path)
    try:
        if eval_flag:
            if evaluation_checkpoint:
                evaluation_checkpoint = PurePath(trainer.log_dir) / "checkpoints" / evaluation_checkpoint
                trainer.test(classifier, dataloader, ckpt_path=(str(evaluation_checkpoint)+".ckpt"))
            else:
                trainer.test(classifier, dataloader, ckpt_path=evaluation_checkpoint)
    except:
        print("Please provide a valid evaluation checkpoint or train the model first.")
