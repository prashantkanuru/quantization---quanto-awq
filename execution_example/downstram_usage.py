from dataclasses import dataclass,field
import sys
import importlib
import typing as t
import transformers
from langchain_core.callbacks import Callbacks

from dataclasses import dataclass
import typing as t
import numpy as np

from tenacity import(
AsyncRetrying,
Retrying,
WrappedFn,
after_log,
retry_if_exception_type,
retry_if_exception_type,
stop_after_attempt,
wait_random_exponential,
)
@dataclass
class CustomRunConfig:
    """A custom Runtime Configuration for timeouts, retries, model path, and device settings for RagasCustomization.
    Will use this for initializing custom llm
    Parameters
    ------------
    timeout: int, optional
        Maximum time (in seconds) to wait for a single operation, by default None.
    max_retries: int, optional
        Maximum number of retry attempts, by default None.
    max_wait: int, optional
        Maximum wait time (in seconds) between retries, by default None.
    max_workers: int, optional
        Maximum number of concurrent workers, by default None.
    exception_types:Union[Type[BaseException],Tuple[Type[BaseException],...]], optional
        Exception types to catch and rety on, by default (Exception)
    log_tenacity: bool, optional
        Whether to log retry attempts using tenacity, by default False.
    seed: int, optional
        Random seed for reproducibility, by default None.
    model_path: str, required
        Path to model on disk.
    device: str, optional
        Device to run the model on, by default 'cuda:0'

    Attributes
    ------------
    rng: numpy.random.Generator
        Random number generator initialized with the specified seed.
        
    """
    model_path:str
    timeout: t.Optional[int]=None
    max_retries:t.Optional[int]=None
    max_wait:t.Optional[int]=None
    max_workers:t.Optional[int]=4
    exception_types:t.Optional[t.Union[t.Type[BaseException],t.Tuple[t.Type[BaseException],...]]] = None
    log_tenacity: bool = True
    seed:t.Optional[int]=None
    device:int=0

    def __post_init__(self):
       self.rng = np.random.default_rng(seed=self.seed) 

       
@dataclass
class QuantizedLLMPipeline:
    """This would be a class that takes in a quantized model -- which is initialized through a
    state-dictionary to be used with transformers pipeline for text generation.

    The workflow for using this class would be to:
    1. First instantiate an object with a pre-initialized quantized model and provide the model string, device torch
    'int' identifier: -1 for CPU and 0 ,... for GPUs and finally -- as RAGAS does not explicitly handle system prompt requirements
    which is a good practice otherwise -- requires us to provide a system prompt
    2. self.initialize_pipeline()
    3. set llm=self.initialize_pipeline()
    4. then use this llm for initializing the faithfulness metric - two ways to do it -- initialize this llm when calling the
    evaluation metric in the evaluate() of 'RAGAS' or create a custom class that is to be called. The latter is preferrable as it
    would reduce the debugging requirements.

    TODO:
    To pick up the device automatically - i.e. the CUDA if torch.cuda.is_available() then check for the count of gpus available
    and also check for device map if provided or else set one using accelerate in such a way that no block is split into two
    different devices.

    ### For faithfulness: generate method has to work for these methods from the class Faithfulness(MetricWithLLM, SingleTurnmetric):

        asyn def _create_verdicts(self, row:t.Dict, statements:t.List[str], callbacks: Callbacks)-> NLIStatementOutput:
    ### Going in Deeper:

    def to_string(self, data:t.Optional[InputModel]=None)-> str:
        return(
            self._generate_instruction()
            +"\n"
            +self._generate_output_signature()
            +"\n"
            +self._generate_examples()
            +"\nNow perform the above instruction with the following input\n"
            +(
            "input: " + data.model_dump_json(indent=4) + "\n"
            if data is not None
            else "input: (None)\n"
            )
            + "ouput: "
        )
    
    """
    model_quantized:transformers.models.phi3.modeling_phi3.Phi3ForCausalLM
    model_hfhub_repo_id_or_directory:str
    device: int
    system_prompt:str = """You are an ai assistant who is adept at following the instructions and examples provided and provide an
    output compliant with the instructions and in the format provided by the examples."""
    temperature:float =0.0
    run_config:CustomRunConfig =field(default_factory= lambda : CustomRunConfig("/home/ubuntu/ssl/ms_phi3mini_128k_model"))
    multiple_completion_supported:bool = False
    
    
    def set_run_config(self, run_config: CustomRunConfig):
        self.run_config = run_config
    
    def __post_init__(self):
        self._ensure_torch()
        self._ensure_auto_tokenizer()

    def _ensure_torch(self):
        #check if torch is imported:
        if 'torch' not in sys.modules:
            print("torch library not found, attempting to import it.")
            try:
                global torch
                torch=importlib.import_module('torch')
                print("successfully imported torch")
            except ImportError:
                raise ImportError("torch is not installed. Please install it using 'pip install torch'")
    def _ensure_auto_tokenizer(self):
        #Check if transformers and AutoTokenizer are already imported.
        if 'transformers' not in sys.modules:
            print("transformers library not found, attempting to import it.")

            try:
                global transformers
                transformers = importlib.import_module('transformers')
                print("Successfully imported transformers")
            except ImportError:
                raise ImportError("transformers library is not installed. Please install it using 'pip install transformers.'")

        try:
            from transformers import AutoTokenizer
            self.AutoTokenizer = AutoTokenizer
            print("AutoTokenizer successfully imported and assigned")

        except AttributeError:
            raise ImportError("Failed to import AutoTokenizer from transformers.")
           

        return self.AutoTokenizer

        
    def initialize_pipeline(self):
        if self.device is None:
            self.device = 0 if torch.cuda.is_available() else -1
            print(f"Using device: {device}")

        #Initialize the tokenizer
        tokenizer = self.AutoTokenizer.from_pretrained(self.model_hfhub_repo_id_or_directory)
        try:
            from transformers import pipeline
            pipe = pipeline("text-generation", model=self.model_quantized, tokenizer=tokenizer, device=self.device)
            return pipe
        except ImportError:
            raise ImportError("Failed to import pipeline from transformers. Please ensure the transformers library is installed.")
            
        #Initialize the pipeline with the quantized model:
        

        #return pipe

    def generate_prompt_message(self):
        pass
    def generate(self, data:str,system_prompt:str=None,
            stop:t.Optional[t.List[str]]=None,
            callbacks:t.Optional[Callbacks]=None,
            n:int=1,
            temperature:t.Optional[float]=None):
        """Trigerring the text generation pipeline and extract out the pipe.
            generation_args={
        "max_new_tokens":25,
        "return_full_text":False,
        "temperature":0.0,
        "do_sample":False,
        }
        output=pipe(messages,**generation_args)
        print(output[0]['generated_text'])

        the above code requires to fall in sync with the following method used in faithfulness:
        Pydantic Prompt driven generate()
        #removed temperature: t.Optional[float]=None, -- as I am using self.temperature
        """
        if temperature is None:
            generation_args={ "max_new_tokens":500,
            "return_full_text":False,
            "temperature":self.temperature,
            "do_sample":False,}
    
        if system_prompt == None:
            message= [{"role":"system","content":self.system_prompt},{"role":"user","content":data},]
        else:
            message= [{"role":"system","content":system_prompt},{"role":"user","content":data},]
        if n==1:
            result=self.initialize_pipeline()(message,**generation_args)
            return result[0]['generated_text']
        else:
            results_collection=[]
            for i in range(n):
                result_i=self.initialize_pipeline()(message,**generation_args)
                results_collection.append(result_i[0]['generated_text'])
            return results_collection