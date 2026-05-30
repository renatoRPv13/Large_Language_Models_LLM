"""
Environment variables used inside the lesson
"""
import os

class LessonEnv:
    """Environment variables used inside the lesson"""
    ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    CONF_PATH = os.path.join(ROOT_DIRECTORY, 'conf.yaml')
