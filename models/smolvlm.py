import torch
from PIL import Image
import os
from typing import Union
import logging
from rich.logging import RichHandler
import time

from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset

MODEL_PATH = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
CACHE_DIR = "hf_cache"
LOG_BASENAME = "smolvlminfer.log"
OUTPUT_DELIM = "Solution:\nAssistant:"

class SmolVLMInfer(object):
    def __init__(self, model_path = MODEL_PATH):
        self.model_path = model_path
        self.processor, self.model = self.load(model_path)
        logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    def load(self, model_path):
        logging.info(f"Loading model + processor from {model_path}")
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=CACHE_DIR)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2"
        ).to("cuda")
        logging.info("Model + processor loaded successfully!")
        return processor, model
    
    def get_response(self, user_prompt: str, decoded_image: Union[Image.Image, None] = None):
        try:
            response = self.infer([self._create_query(user_prompt, decoded_image)])
        except Exception as e:
            logging.error(e)
            return ""
        return response

    def infer(self, messages):
        logging.debug(f"Infer: {messages}")  # Log inference start
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return self._parse_answer(generated_texts[0])
  
    def _create_query(self, user_prompt, decoded_image):
        logging.debug(f"Create Query: {user_prompt}")
        if decoded_image is None:
            logging.error(f'You are using a model that supports vision: {self.model}, but no image was provided when generating a response. This is likely unintended.')
        message = {
            "role": "user",
            "content": [
                {"type": "image", "image": decoded_image},
                {"type": "text", "text": user_prompt},
            ]
        }   
        return message

    def _parse_answer(self, response):
        delim_pos = response.rfind(OUTPUT_DELIM)
        if delim_pos == -1:
            logging.error(f'Invalid output delimiter \"{OUTPUT_DELIM}\" used.')
            return ""

        return response[delim_pos + len(OUTPUT_DELIM):].strip()
