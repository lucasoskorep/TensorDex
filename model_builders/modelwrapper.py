from collections import Callable


class ModelWrapper(object):
    def __init__(self, model_func:Callable, model_preprocessor:Callable):
        self.model_func = model_func
        self.model_preprocessor = model_preprocessor
