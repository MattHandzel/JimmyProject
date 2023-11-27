import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def add_to_amino_acids(a_sequence: str, amino_acids: list):
    for acid in a_sequence:
        if acid not in amino_acids:
            amino_acids.append(acid)

def feature_map(p_sequence, label_encoder: LabelEncoder):
    return [tf.one_hot(label_encoder.transform(list(x)), len(label_encoder.classes_)) for x in p_sequence]

def convert_to_space_separated_string(series):
    return ' '.join(series)

def construct_sentences(dataframe):
    cdr3_sentences = dataframe["cdr3"]
    epitope_sentences = dataframe["antigen.epitope"]
    return cdr3_sentences, epitope_sentences

def pad_sentences(sentences, max_length, tokenizer : BertTokenizer):
    input_ids = []
    attention_masks = []

    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.stack(input_ids), torch.stack(attention_masks)

def shuffle_data(data, random_state=42):
    np.random.seed(random_state)
    np.random.shuffle(data)
    return data

def train_test_split_no_shuffle(data, test_size):
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]









