import logging
import sys


def get_instance():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%m-%d-%Y %H:%M:%S"
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("debug.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger


class LOGGER:
    __instance = None

    @staticmethod
    def get_instance():
        """Static access method."""
        if LOGGER.__instance == None:
            LOGGER.__instance = get_instance()
        return LOGGER.__instance

    def __init__(self):
        self.logging = LOGGER.get_instance()
