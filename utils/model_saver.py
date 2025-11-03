from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from quanto import quantize, freeze, qint8, qint4, qint2, qfloat8
from pydantic import BaseModel, Field, root_validator,model_validator
import torch
from safetensors.torch import save_file
import logging
from typing import Dict, Union

#Persist the model on to the disk:
#TODO: Turn this into a Mixin
class BaseAncillary(BaseModel):
    """This Base Class is being called an Ancillary class because it is 
    only meant for loading the model from huggingface hub using, snapshot_download.

    Args:

    model_id: Huggingface_Hub repo id ex. meta/llama3
    target_directory: the directory on the local disk where the huggingface_hub repo id will
    be replicated.

    This class also does the type validation and also checks for the existence of the 
    directory in the target_directory provided path.

    Methods:
    load_the_model: A class method to load the model from huggingface hub into the
    target directory

    Example:

    load_the_model=BaseAncillary(repo_id="meta/llama3",target_directory="~/path_to_target")
    BaseAncillary.load_model(load_the_model)
    
    """
    
    model_id:str = Field(..., description="The repo_id of model from huggingface hub\
     this is a required field ")
    target_directory:str = Field(..., description="The path of the directory where it has to\
    be stored")

    @model_validator(mode='before')
    #basic aim is to check if the arguments are strings or not and if the target_directory
    #path exists or not!
    def check_values(cls,values):
        '''check if 'model_id' is a string and 'target_directory' path exists'''    
        if not isinstance(values.get('model_id'),str):
            raise ValueError('model_id must be a string')
        if not isinstance(values.get('target_directory'),str):
            raise ValueError('target_directory must be a string')
        #if not os.path.exists(values.get('target_directory')):
            #raise ValueError('target_directory does not exist')
        return values

    @classmethod
    def load_the_model(cls,instance):
        '''Loads the entire model repo from huggingface hub using snapshot_download'''

        snapshot_download(repo_id=instance.model_id, local_dir=instance.target_directory)
    class Config:
        protected_namespaces=()