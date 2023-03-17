import os
import pickle
import numpy as np
import torch

from gpt_model import *
print('Loading: ', model_finetuned_pt_file)
model.load_state_dict(torch.load(model_finetuned_pt_file))

test_cases = [
    '0',
    'как говорит радио Москвы',
    'Добрая жена ПП=',
    'Старый конь ПП=',
    'Куда ни глянь везде ПФ=',
    'Как не стоит тестировать ПФ=',
    'acknowledgementПФ='
]

for sent in test_cases:
    output = inference(sent)
    print("Prompt : ", sent)
    print("Predict: ", output)
    print("="*80)