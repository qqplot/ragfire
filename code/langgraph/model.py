import argparse
import gc
import multiprocessing as mp
import os
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from accelerate import init_empty_weights
from timer import timers
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM, 
                          BloomForCausalLM, OPTForCausalLM, LlamaForCausalLM, BitsAndBytesConfig
                        )
from transformers.integrations import HfDeepSpeedConfig

from utils import (GB, add_model_hooks, cache_bytes,
                   get_filename, get_quant_config, hidden_bytes, meta_to_cpu,
                   model_bytes, write_benchmark_log)
from packaging import version

assert version.parse(deepspeed.__version__) >= version.parse("0.10.3"), "ZeRO-Inference with weight quantization and kv cache offloading is available only in DeepSpeed 0.10.3+, please upgrade DeepSpeed"

def load_model(model_name: str):
    if model_name == "exaone":
        model_path = f"/home/shared/RAG/model/{model_name}"

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        quantization_config = BitsAndBytesConfig(
            # load_in_4bit=True, 
            # bnb_4bit_compute_dtype=torch.float16 
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            quantization_config=quantization_config,
        ).eval()

    elif model_name = "qwen":
        model_path="/home/shared/RAG/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
        pass


    return model, tokenizer