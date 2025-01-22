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

import os
import fire
import json
import torch
import pickle
import regex as re
import ast
from PIL import Image
import jsonlines
from tqdm import tqdm
from loguru import logger
import pandas as pd
from typing import List, Dict, Union
from models.ursa_model import UrsaProcessor, UrsaForTokenClassification

PROMPT = 'You are given a problem and a step-by-step solution. You need to check the correctness of each step.\nQuestion:'

def return_score(scores: torch.Tensor, operation: str):
    if operation == 'min':
        if scores.numel() == 0:
            return torch.tensor([0.])
        scores = scores.view(-1)
        return torch.min(scores)
    elif operation == 'avg':
        if scores.numel() == 0:
            return torch.tensor([0.])
        scores = scores.view(-1)
        return torch.mean(scores)

def replace_specific_plus_minus_with_ki(text):
    pattern = r'Step \d+'
    matches = list(re.finditer(pattern, text))
    positions = [(match.start(), match.end()) for match in matches]
    
    text_list = list(text)
    insert_pos = []
    try:
        for i in range(1, len(positions)):
            for j in range(positions[i][0] - 1, positions[i - 1][1], -1):
                if text_list[j] != ' ' and text_list[j] != '\n':
                    insert_pos.append(j + 1)
                    break
        
        answer_start = text.find('†Answer:')
        for j in range(answer_start - 1, positions[-1][1], -1):
            if text_list[j] != ' ' and text_list[j] != '\n':
                insert_pos.append(j + 1)
                break
        for index in sorted(insert_pos, reverse=True):
            text = text[:index] + ' и' + text[index:]
        return text
    except:
        return text + ' и'

def extract_answer_try_all_methods(s, remove_parentheses=True):
    s = s.replace('[UNUSED_TOKEN_145]', '')
    if not s:
        return s
    last_plus_index = s.rfind("†")
    if last_plus_index != -1:  # "+" found in the string
        return s[last_plus_index+1:]
    
    matches = re.findall(r"(†Answer:|†Answer|answer:|Answer:|therefore,|so,|Therefore,|So,|Thus,|thus,)([\s\S]*?)(?=answer:|Answer:|therefore,|so,|Therefore,|So,|Thus,|thus,|$)", s)
    if matches:
        ans = matches[-1][1].strip()
        return ans.strip()

    return s.strip() # unsuccessful, return total response

def prepare_input(question, response):
    if isinstance(question, float):
        instruction = PROMPT + '\n' + response
    else:
        instruction = PROMPT + question + '\n' + response
    instruction = replace_specific_plus_minus_with_ki(instruction)
    return instruction

def single_inference(
    processor: UrsaProcessor,
    model: UrsaForTokenClassification,
    cuda_device: int,
    input_prompt: str = '',
    image: Union[str, Image.Image] = None,
    system_prompt: str = 'You are a helpful assistant.',
    **kwargs
) -> str:
    conv = [{"role": "system", "content": system_prompt}] if system_prompt else []
    conv.append({
        "role": "user",
        "content": "<|image|>" + input_prompt
    })
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    if isinstance(image, str):
        raw_image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        raw_image = image.convert('RGB')
    else:
        raw_image = None
    raw_image = [raw_image]
    inputs = processor(prompt, raw_image, return_tensors='pt').to(cuda_device, torch.bfloat16)
    tag_id = processor.tokenizer.encode(' и', add_special_tokens=False)
    with torch.no_grad():
        reward = model(**inputs).logits
        input_ids = inputs['input_ids'].view(-1)
        insert_values = torch.full((575,), -1).to(input_ids.device)
        input_ids = torch.cat((input_ids[:1], insert_values, input_ids[1:]))
        reward = reward.view(-1)[input_ids == tag_id[0]]
        reward = torch.sigmoid(reward).view(-1)  
        min_score = return_score(reward, 'min')
        avg_score = return_score(reward, 'avg')
    return min_score, avg_score

def split_array_projection(array, x, l):
    n = len(array)
    base_length = n // x
    k, m = divmod(base_length, l)
    split_indices = [(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(l)]
    projected_indices = [(start * x, end * x) for start, end in split_indices]
    result = [array[start:end] for start, end in projected_indices]
    
    return result


def inference(
    dataset_name: str,
    data_path: str,
    model_path: str,
    dtype: torch.dtype,
    output_path: str,
    image_root: str,
    cuda_device: int,
    cuda_sum: int,
    best_of_n: int,
    *args,
    **kwargs
):
    if 'qwen' not in model_path.lower():
        logger.warning("'qwen' not in model_path, make sure you have selected the correct qwen2-math-vl model")

    logger.info('Loading data...')
    data = []
    print('{}: {}'.format('name', dataset_name))
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif data_path.endswith('.jsonl'):
        with open(data_path, 'r', encoding='utf-8') as fp:
            for item in jsonlines.Reader(fp):
                data.append(item)
    elif data_path.endswith('csv'):
        df = pd.read_csv(data_path)
        for k in ['model_answer', 'extraction']:
            df[k] = df[k].apply(ast.literal_eval)
        num_n = len(df.iloc[0, :]['extraction'])
        for i in range(df.shape[0]):
            for j in range(num_n):
                data.append({
                    'input': prepare_input(df.iloc[i, :]['question'], df.iloc[i, :]['model_answer'][j]),
                    'image': os.path.join(image_root, df.iloc[i, :]['image']),
                    'label': df.iloc[i, :]['label{}'.format(j)]
                })
    elif any(data_path.endswith(suffix) for suffix in ['pk', 'pkl']):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    assert len(data) % best_of_n == 0
    chunks = split_array_projection(data, best_of_n, cuda_sum)
    data = chunks[cuda_device]
    logger.info('Loading model...')
    model: UrsaForTokenClassification = UrsaForTokenClassification.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(cuda_device)
    processor: UrsaProcessor = UrsaProcessor.from_pretrained(model_path)
    print(args)
    inference_dataset(processor, model, cuda_device, data, image_root=image_root, output_path=output_path, best_of_n=best_of_n, *args, **kwargs)

def inference_dataset(
    processor: UrsaProcessor,
    model: UrsaForTokenClassification,
    cuda_device: int,
    data: List[Dict],
    image_root: str,
    output_path: str,
    best_of_n: int,
    *args,
    **kwargs
):
    res_min = []
    res_avg = []
    for i, d in tqdm(enumerate(data), total=len(data), desc=f'[Device: {cuda_device}]'):
        image_path = data[i]['image']
        min_reward, avg_reward = single_inference(processor, model, cuda_device, input_prompt=data[i]['input'], image=image_path, do_sample=False)
        res_min.append(min_reward.item())
        res_avg.append(avg_reward.item())    
    
    avg_scores = torch.tensor(res_avg).view(-1, best_of_n).float()
    min_scores = torch.tensor(res_min).view(-1, best_of_n).float()
    labels = [d['label'] for d in data]
    labels = torch.tensor(labels).view(-1, best_of_n).int()
    
    for top in [8, 16, 32, 64]:
        sub_score = avg_scores[:, :top]
        sub_min_score = min_scores[:, :top]
        _, max_indices = torch.max(sub_score, dim=1)
        _, max_min_indices = torch.max(sub_min_score, dim=1)
        corresponding_values = labels[torch.arange(labels.size(0)), max_indices]
        corresponding_min_values = labels[torch.arange(labels.size(0)), max_min_indices]
        torch.save(corresponding_values, output_path.replace('.pt', '_{}_sampling{}.pt'.format('avg', top)))
        torch.save(corresponding_min_values, output_path.replace('.pt', '_{}_sampling{}.pt'.format('min', top)))




if __name__ == '__main__':
    fire.Fire(inference)
