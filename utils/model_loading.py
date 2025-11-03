from quanto import quantize
from pydantic import BaseModel, Field, root_validator
from transformers import AutoModelForCausalLM
import torch
from torch import device as torch_device
from safetensors.torch import safe_open
import logging
import os
from typing import Optional
#Configure logging:
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    handlers=[
        logging.FileHandler("error.log"),
        logging.StreamHandler()
    ]
)

#Get the logger
logger = logging.getLogger(__name__)

class LoadQuantizedModel(BaseModel):
    """Base class to load a qunantized models state dictionary.

    """
    model_dir_or_directory:str = Field(..., description="This provides the access to directory where the model\
                          is stored on the disk, this field is required\
                                       this is essentially the model repo dumped from HF HUB")
    state_dict_quantized_path_bin_file: str = Field(..., description="This provides the path to the\
                                           state dictionary of the quanto quantized pytorch\
                                           model as .bin file")
    
    #quantization_types=['qint2','qint4','qint8','qfloat8']
    @model_validator(mode='before')
    def check_values(cls,values):
        '''Checks for Data Type Validation'''
        if not isinstance(values.get('model_dir_or_directory'),str) and not isinstance(values.get('model_dir_or_directory'),os.PathLike):
            raise ValueError('model_id must be a string\
                             and a os.PathLike i.e. the model repo should exist in the\
                             provided directory path') 
        if not isinstance(values.get('state_dict_quantized_path_bin_file'),str) and not isinstance(values.get
                                                                                    ('state_dict_quantized_path_bin_file'),os.PathLike ) and values.get('state_dict_quantized_path_bin_file').split('.')!='bin':
            raise ValueError("state_dictionary path should be a string and also\
                             os.PathLike, i.e. the file should exist at the given path,\
                             which is not the case now\
                             or else it should be a .bin file")
        
        return values
    
    #Creating global variables:
    

    
    @classmethod
    def load_quantized_pytorch_bin(cls, instance)->torch.nn.Module:
        '''Loads the qunatized bin file

        Why the context manager of torch.device('meta'):
        When the checkpoint are saved with torch.save, tensors are tagged with the device they are saved on. With 'torch.load'
        tensor storages will be loaded to the device they are tagged with (unless this behavior is overridden using the 'map_location'
        flag)

        We can use the torch.device() context manager with device='meta' when we instantiate the nn.Module()
        The torch.device() context manager makes sure that factory calls will be performed as if they were passed the specified 'device'
        as an argument. Tensors on `torch.device('meta') do not carry data. However, they possess all other metadata a tensor carries
        such as .size(), .stride(),.requires_grad(), and others.

        Double click:
        `m=SomeModule(1000)`

        This allocates memory for all parameters/buffers and initializes them per default initialization schemes defined in 
        `SomeModule.__init__()` , which is wasteful when we want to load a checkpoint for the following reasons:
        - The result of the initialization kernels will be overwritten 
        
        
        is loaded with 'torch.load'

        Another approach to this is:
        meta = torch_device('meta')
        cpu=torch_device('cpu')
        gpu= torch_device('cuda:0')

        with meta:
            model=AutoModelForCausalLM.from_pretrained('repo_id',torch_dtype='auto')
            quantize(model)
        model.to_empty(device=cpu)
        state_dict

        ## What does model.to_empty mean:
        - https://pytorch.org/tutorials/prototype/skip_param_init.html
        - Skipping Module Parameter Initialization:
        Introduction: When a module is created, its learnable parameters are initialized according to a default initialization 
        scheme associated with the module type. For example, the weight parameter for a torch.nn.Linear module is 
        initialized from a Uniform(-1/sqrt(in_features),1,sqrt(in_features)) distribution. If some other initialization scheme is desired
        this has traditionally required re-initializing the parameters after module instantiation.

        ```
        from torch import nn
        #Initializes weight from the default distribution Uniform(-1/sqrt(in_features),1,sqrt(in_features))
        m=nn.Linear(10,5)
        #Re-initialize weight from different distribution:
        nn.init.orthogonal_(m.weight)
        ```
        In this case, the initialization done during construction is wasted computation, and it may be non-trivial if the weight
        parameter is large.

        Skipping Initialization:
        It is now possible to skip parameter initiali
        '''
        meta=torch_device('meta')
        cpu=torch_device('cpu')
        gpu=torch_device('cuda:0')    
        try:
            with meta:
                model=AutoModelForCausalLM.from_pretrained(instance.model_dir_or_directory)
                quantize(model)
                

        except Exception as e:
            logger.error(f"Failed to load model from {instance.model_dir} without\
                         trust_remote_code. Error: {e}")
            try:
                with meta:
                    model=AutoModelForCausalLM.from_pretrained(instance.model_dir_or_directory, trust_remote_code=True)
                    quantize(model)
            except Exception as e:
                logger.error(f"Failed to load model from {instance.model_dir} with trust_remote_code\
                             . Error: {e}")
                raise ValueError(f"Unable to load model from directory: {instance.model_dir}")

        
        #quantize(model, weights=weights, activations=instance.activation_quantization)
        #model.to(torch_device('meta'))
        #quantize(model)
        model.to_empty(device=cpu)
        state_dictionary=torch.load(instance.state_dict_quantized_path_bin_file)
        state_dictionary=model.load_state_dict(state_dictionary,assign=True)
        return model

    class Config:
        protected_namespaces=()
    @classmethod
    def load_quantized_pytorch_safetensor(cls, instance)->dict:
        '''Loads the qunatized bin file 
        '''
        try:
            with meta:
                model=AutoModelForCausalLM.from_pretrained(instance.model_dir_or_directory)
                quantize(model)

        except Exception as e:
            logger.error(f"Failed to load model from {instance.model_dir_or_directory} without\
                         trust_remote_code. Error: {e}")
            try:
                with meta:
                    model=AutoModelForCausalLM.from_pretrained(instance.model_dir_or_directory, trust_remote_code=True)
                    quantize(model)
            except Exception as e:
                logger.error(f"Failed to load model from {instance.model_dir_or_directory} with trust_remote_code\
                             . Error: {e}")
                raise ValueError(f"Unable to load model from directory: {instance.model_dir_or_directory}")

        
        #quantize(model, weights=weights, activations=instance.activation_quantization)
        #model.to(torch_device('meta'))
        #quantize(model)
        state_dictionary=safe_open(instance.state_dict_quantized_path_safetensors_file)
        state_dictionary=model.load_state_dict(state_dictionary,assign=True)
        return model