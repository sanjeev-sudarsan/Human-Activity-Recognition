import logging
import tensorflow as tf


def set_loggers(paths=None, logging_level=0, b_stream=False, b_debug=False):

    # std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # tf logger
    logger_tf = tf.get_logger()
    logger_tf.setLevel(logging_level)

    evaluation_logger = logging.getLogger("evaluation")
    evaluation_logger.setLevel(logging_level)
    evaluation_logger.addHandler(hdlr=logging.FileHandler(paths["file_evaluation_log"]))

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    if b_debug:
        tf.debugging.set_log_device_placement(False)
