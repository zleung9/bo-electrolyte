import os
import logging
import gc
import torch

def clean_gpu():
   gc.collect()
   torch.cuda.empty_cache()

def create_logger(logger_name, log_path = None, append = False, simple_fmt=False):
    
    if not append and os.path.isfile(log_path):
        with open(log_path, 'w') as f: pass
    # If append is False and the file exists, clear the content of the file.

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if log_path is None:
        handler = logging.StreamHandler() # show log in console
    else:
        handler = logging.FileHandler(log_path) # print log in file
    
    handler.setLevel(logging.DEBUG)
    if simple_fmt:
        handler.setFormatter(
            logging.Formatter(
                fmt = "%(message)s"
            )
        )
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt = '%(asctime)s %(levelname)s:  %(message)s',
                datefmt ='%m-%d %H:%M'
            )
        )
    logger.addHandler(handler)

    return logger


class Parameters():
    
    """
    A parameter object that maps all dictionary keys into its name space.
    The intention is to mimic the functions of a namedtuple.
    """
   
    def __init__(self, parameter_dict):
        
        # "__setattr__" method is changed to immutable for this class.
        super().__setattr__("_parameter_dict", parameter_dict)
        self.update(parameter_dict)
        

    def __setattr__(self, __name, __value):
        """
        The attributes are immutable, they can only be updated using `update` method.
        """
        raise TypeError('Parameters object cannot be modified after instantiation')


    def get(self, key, value):
        """
        Override the get method in the original dictionary parameters.
        """
        return self._parameter_dict.get(key, value)
    

    def update(self, parameter_dict):
        """
        The namespace can only be updated using this method.
        """
        self._parameter_dict.update(parameter_dict)
        self.__dict__.update(self._parameter_dict) # map keys to its name space

    def to_dict(self):
        """
        Return the dictionary form of parameters.
        """
        return self._parameter_dict 


    @classmethod
    def from_yaml(cls, config_file_path):
        """
        Load parameter from a yaml file.
        """
        import yaml

        with open(config_file_path) as f:
            trainer_config = yaml.full_load(f)

        return Parameters(trainer_config)

if __name__ == "__main__":
    config = Parameters.from_yaml("./experiments/RGPE_config.yaml")
    pass