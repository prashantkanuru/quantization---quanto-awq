from quanto import quantize, freeze, qint8, qint4, qint2, qfloat8
from pydantic import BaseModel, Field, root_validator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import device as torch_device
from safetensors.torch import save_file
import logging
from typing import Dict, Union,Literal,ClassVar
import os
from enum import Enum
#Configure logging:
#TO DO: shift the logging part/logger to a different file and call from there
#TO DO: I also want to log in every successful execution of the try block
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

#torch.cuda.is_available()

"""
class Device(Enum):
    cpu=torch_device('cpu')
    meta=torch_device('meta')
    gpu=torch_device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""
cpu=torch_device('cpu')
meta=torch_device('meta')
gpu=torch_device('cuda:0' if torch.cuda.is_available() else 'cpu')

class QuantizeUsingQuanto(BaseModel):
    """BaseQuantization Model for using Quanto to quantize a Pytorch based LLM
       This can be considered to be a wrapper around quanto and save the quantized
       model to disk or persist to the disk and use it again to do text generation. 
    """
    #device: Device =Field(..., description="the device on to which the pytorch nn.Module subclass i.e. the pytorch\
    #modules are loaded or tensors are loaded")
    model_dir_or_directory:str = Field(..., description="This provides the access to directory where the model\
                          is stored on the disk, this field is required")
    #THe below data type instantiation to Literal is done because of the error:
    #Pydantic UserError: A non-annotated attribute was detected: quantization_types=['qint2','qint4','qint8','qfloat8'].
    #
    # PydanticUserError: A non-annotated attribute was detected: `quantization_types = ['qint2', 'qint4', 'qint8', 'qfloat8']`. All model fields require a type annotation; if `quantization_types` is not meant to be a field, you may be able to resolve this error by annotating it as a `ClassVar` or updating `model_config['ignored_types']`.
    #For further information visit https://errors.pydantic.dev/2.7/u/model-field-missing-annotation
    quantization_type:Literal['qint2','qint4','qint8','qfloat8'] =Field(..., description="type of quantization, choose from\
                                  'qint8','qint4',qint2")
    activation_quantization:Literal['qint8','qfloat8',"None"] = Field(..., description="States whether we intend to\
                                         calibrate the activation function for\
                                         quantization")
    quantization_types: ClassVar = ['qint2','qint4','qint8','qfloat8']
    
    @model_validator(mode='before')
    def check_values(cls,values):
        '''Checks for Data Type Validation'''
        if not isinstance(values.get('model_dir_or_directory'),str):
            raise ValueError('model_id must be a string') 
        if values.get('quantization_type') not in cls.quantization_types:
            raise ValueError(f"Quantization_type must be one of {cls.quantization_type}")
        return values
    @classmethod
    def quantization(cls, instance)->Dict[str,Union[torch.Tensor,str]]:
        '''quantization using the quanto library and capture the
        model's state dictionary or checkpoints.
        '''
        try:
            model=AutoModelForCausalLM.from_pretrained(instance.model_dir_or_directory)
            #model.to(torch_device(gpu))
            logger.info(f"loaded the model successfully from {instance.model_dir_or_directory}")

        except Exception as e:
            logger.error(f"Failed to load model from {instance.model_dir_or_directory} without\
                         trust_remote_code. Error: {e}")
            try:
                model=AutoModelForCausalLM.from_pretrained(instance.model_dir_or_directory, trust_remote_code=True)
                #model.to(torch_device(gpu))
                logger.info(f"Successfully loaded the model from {instance.model_dir_or_directory} using trust_remote_code")
            except Exception as e:
                logger.error(f"Failed to load model from {instance.model_dir} with trust_remote_code\
                             . Error: {e}")
                raise ValueError(f"Unable to load model from directory: {instance.model_dir_or_directory}")

        
        weights_dict={'qint2':qint2,'qint4':qint4,'qint8':qint8}
        weights=weights_dict.get(instance.quantization_type,qfloat8)

        activations_weights={'qint8':qint8,'qfloat8':qfloat8,"None":None}
        activations=activations_weights.get(instance.activation_quantization, None)
        
        quantize(model, weights=weights, activations=activations)
        freeze(model)
        state_dictionary=model.state_dict()
        return state_dictionary

    @classmethod
    def save_quantized_to_pytorch_bin(cls,state_dictionary:Dict[str,Union[torch.Tensor, str]],path_to_bin_file:Union[str,os.PathLike]):
        '''serialize the model's state-dictionary serialized to a .bin file.
        This is the practice in huggingface hub - that a PyTorch model's weights are
        either stored as a .bin file or as .safetensors file.

        Arguments:
        state_dictionary: model's state dictionary aka model.state_dict()
        path_to_bin_file: aka ~/path_to_directory/pytorch_model_state_dict.bin

        Returns:

            None
        '''
        try:

            torch.save(state_dictionary,path_to_bin_file)
        except Exception as e:
            print("Failed to serialize the file to a .bin file")
            raise
        return None
    @classmethod
    def save_quantized_to_safetensor(cls, state_dictionary:Dict[str,Union[torch.Tensor,str]], path_to_safetensors_file:Union[str,os.PathLike]):
        '''serialize to a safetensors file aka .safetensor
        
        Arguments:
        state_dictionary: model's state dictionary aka model.state_dict()
        path_to_safetensors_file: aka ~/path_to_directory/pytorch_model_state_dict.safetensor
        
        Returns:
            None
        '''
        try:
            #Separate tensors and metadata
            tensors={}
            metadata={}
            for name,value in state_dictionary.items():
                if isinstance(value, torch.Tensor):
                    tensors[name]=value
                else:
                    metadata[name]=value
            save_file(tensors,path_to_safetensors_file,metadata=metadata)
            print("model's state dictionary has been serialized as a safetensor file")
        
        except Exception as e:
            print(f"Failed to serialize to safetensor file: {e}")
            raise
    class Config:
        protected_namespaces=()