from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
import torch


class LLMModel:
    def __init__(self, model_name: str = 'mistralai/Mixtral-8x7B-Instruct-v0.1', 
                 quantized: bool = True, device: int = 2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if quantized:
            # https://github.com/huggingface/blog/blob/main/mixtral.md#load-mixtral-with-4-bit-quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            # https://github.com/yhoshi3/RaLLe/blob/main/scripts/configs/base_settings/llms.json#L25
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": device},
                                                              # device_map="auto",
                                                              quantization_config=quantization_config)
        else:
            # https://stackoverflow.com/questions/77237818/how-to-load-a-huggingface-pretrained-transformer-model-directly-to-gpu
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            
        self.model.eval()

        # https://github.com/yhoshi3/RaLLe/blob/main/ralle/llms/initialize.py#L85
        # pipeline_args are again in: https://github.com/yhoshi3/RaLLe/blob/main/scripts/configs/base_settings/llms.json#L35
        self.pipe = pipeline("text-generation", model=self.model, 
                             tokenizer=self.tokenizer,
                             max_new_tokens=32,
                             top_k=50,
                             do_sample = False,
                             # https://huggingface.co/docs/transformers/en/generation_strategies
                             # temperature=0.000000001, 
                             # temperature=0, # tried with uncomment # do_sample = False,
                             # UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`. 
                             # temperature=0, # tried with # do_sample = True,
                             # ValueError: `temperature` (=0.0) has to be a strictly positive float, otherwise your next token scores will be invalid. If you're looking for greedy decoding strategies, set `do_sample=False`.
                             repetition_penalty=1.204819277108434,
                             return_full_text=False, # https://www.pinecone.io/learn/mixtral-8x7b/
                            )

    def generate_text(self, input_prompt: str):
        # https://huggingface.co/docs/transformers/main/en/tasks/prompting
        torch.manual_seed(0)
        # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
        return self.pipe(input_prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)[0]["generated_text"]

