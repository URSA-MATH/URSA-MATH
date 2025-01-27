# Copyright (2025) Bytedance Ltd. and/or its affiliates 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
import jsonlines
import fire
from typing import List, Dict, Tuple
import json
from concurrent.futures import ThreadPoolExecutor
import re
import os
from tqdm import tqdm
import torch
from PIL import Image

prompt = 'you are given a math problem image, please solve the problem step by step. When you get an answer, please return the correspond option instead of the text content.\nQuestion:'
template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|image|>{}<|im_end|>
<|im_start|>assistant
"""


import re

def extract_answer_try_all_methods(text: str) -> str:
    # Define a list of patterns to match possible answer formats
    patterns = [
        r'Answer: \\boxed\{(.*?)\}',
        r'\\boxed\{(.*?)\}',
        r'†Answer:(.*?)(?:\n|$)',
        r'†Answer: (.*?)(?:\n|$)',
        r'†Answer: \n(.*?)(?:\n|$)',
        r'correct answer：(.*?)(?:\n|$)',
        r'Correct answer：(.*?)(?:\n|$)',
        r'Answer：(.*?)(?:\n|$)',
        r"correct answer is:(.*?)(?:\n|$)",
        r"correct answer is:\n(.*?)(?:\n|$)",
        r"correct answer is:\n\n(.*?)(?:\n|$)",
        r"correct answer is:\n\n\n(.*?)(?:\n|$)",
        r'correct answer:(.*?)(?:\n|$)',
        r'Correct answer:(.*?)(?:\n|$)',
        r'correct answer is(.*?)(?:\n|$)',
        r'Answer:(.*?)(?:\n|$)',
        r'†(.*?)(?:\n|$)',
        r'Answer: (.*?)(?:\n|$)',
        r'The answer is (.*?)(?:\n|$)'
    ]

    # Iterate through all patterns
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Filter out invalid matches (empty strings or single non-alphanumeric characters)
            valid_matches = [
                match.strip() for match in matches
                if match.strip() and (len(match.strip()) > 1 or match.strip().isalnum())
            ]
            if valid_matches:
                # If it's the first pattern, return all valid matches joined by newline
                if pattern == patterns[0]:
                    return '\n'.join(valid_matches)
                # Otherwise, return the first valid match
                return valid_matches[0]

    # If no pattern matches, return the original text
    return text


def mathv_option_trans(x):
    options = x['options']
    if len(options) == 0:
        return ''
    res = []
    for i, image in enumerate(options):
        option_prefix = chr(ord('A') + i)  # A, B, C, ...
        res.append(f"{option_prefix}. {image}")
    return ' '.join(res)

def prepare_data(dataset: str, data_path: str, image_root: str) -> Tuple[List[Dict], List[Dict]]:
    origin = []
    if dataset == 'mathverse':
        origin = json.load(open(data_path, 'r'))
        input_data = []
        for x in tqdm(origin):
            if x['image'] == '':
                input_data.append({
                    'prompt': template.format(x['query_cot']),
                })
            else:
                input_data.append({
                    'prompt': template.format(x['query_cot']),
                    'multi_modal_data': {
                        'image': Image.open(os.path.join(image_root, x['image'])).convert("RGB")
                    }
                })
    elif dataset == 'mathvista':
        input_data = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            for item in tqdm(jsonlines.Reader(fp)):
                input_data.append({
                    'prompt': template.format(prompt + item['question']),
                    'multi_modal_data': {
                        'image': Image.open(os.path.join(image_root, item['image'])).convert("RGB")
                    }
                })
                origin.append(item)
    elif dataset == 'wemath':
        input_data = []
        origin = json.load(open(data_path, 'r'))
        input_data = [
            {
                'prompt': template.format('{}{} {}'.format(prompt, x['question'], x['option'])),
                'multi_modal_data': {
                    'image': Image.open(os.path.join(image_root, x['image_path'])).convert("RGB")
                }
            }
            for x in tqdm(origin)
        ]
    elif dataset == 'mathvision':
        input_data = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            for item in tqdm(jsonlines.Reader(fp)):
                op, question = mathv_option_trans(item), item['question']
                if op != '':
                    question = '{} {}'.format(question, mathv_option_trans(item))
                input_data.append({
                    'prompt': template.format(prompt + question),
                    'multi_modal_data': {
                        'image': Image.open(os.path.join(image_root, item['image'])).convert("RGB")
                    }
                })
                origin.append(item)
    elif dataset == 'dynamath':
        input_data = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            for item in tqdm(jsonlines.Reader(fp)):
                input_data.append({
                    'prompt': template.format(prompt + item['question']),
                    'multi_modal_data': {
                        'image': Image.open(os.path.join(image_root, os.path.join(item['image_dir'], item['image']))).convert("RGB")
                    }
                })
                origin.append(item)
    elif dataset == 'chartqa':
        input_data = []
        origin = json.load(open(data_path, 'r'))
        input_data = [
            {
                'prompt': template.format('{} {}'.format(prompt, x['query'])),
                'multi_modal_data': {
                    'image': Image.open(os.path.join(image_root, x['imgname'])).convert("RGB")
                }
            }
            for x in tqdm(origin)
        ]
    return input_data, origin

def run_infer(
    model: str,
    dataset: str,
    data_path: str,
    output_file: str,
    image_root: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    num_return_sequences: int = 1,
):
    print("Temerature: {}, n_seq: {}".format(temperature, num_return_sequences))
    sample_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=num_return_sequences)
    input_data, origin_data = prepare_data(dataset, data_path, image_root)
    llm = LLM(model=model, tensor_parallel_size=1)
    outputs = llm.generate(
        input_data,
        sampling_params=sample_params,
    )

    new_items = []
    for i, o in enumerate(outputs):
        generated_text = [o.outputs[j].text for j in range(num_return_sequences)]
        new_items.append({
            **origin_data[i],
            'model_answer': generated_text,
            'extraction': [extract_answer_try_all_methods(kk) for kk in generated_text]
        })
    with open(output_file, 'w', encoding='utf-8') as f:
        writer = jsonlines.Writer(f)
        for item in new_items:
            writer.write(item)


if __name__ == "__main__":
    fire.Fire(run_infer)
