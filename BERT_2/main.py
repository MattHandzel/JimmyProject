import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import matplotlib.pyplot as plt
import time
import pickle
from transformers import *
from dataloader import *
from model import *
from early_stopper import EarlyStopper
import pickle

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positives = pd.read_csv("./positive.csv")
    negatives = pd.read_csv("./negative.csv")
    amino_acids = extract_amino_acids(positives)

    amino_acid_label_encoder = LabelEncoder()
    amino_acid_label_encoder.fit(amino_acids)

    all_amino_acids = amino_acid_label_encoder.transform(amino_acids)

    data_cd3r = feature_map(positives["cdr3"], amino_acid_label_encoder)
    data_epitope = feature_map(positives["antigen.epitope"], amino_acid_label_encoder)

    tokenizer = BertTokenizer(vocab_file="./amino_acid_vocab.txt")

    max_length = 32

    cdr3_sequence_epitope_encoder = LabelEncoder()
    cdr3_sequence_epitope_encoder.fit(positives["antigen.epitope"])

    cdr3_labels = cdr3_sequence_epitope_encoder.transform(positives["antigen.epitope"])
    for column in positives.columns:
        positives[column] = positives[column].apply(convert_to_space_separated_string)

    cdr3_sentences, epitope_sentences = construct_sentences(positives)

    shuffle_data(cdr3_sentences, 42)
    shuffle_data(epitope_sentences, 42)
    shuffle_data(cdr3_labels, 42)

    cdr3_input_ids, cdr3_attention_masks = pad_sentences(
        cdr3_sentences, max_length, tokenizer
    )
    epitope_input_ids, epitope_attention_masks = pad_sentences(
        epitope_sentences, max_length, tokenizer
    )

    cdr3_combined = torch.cat((cdr3_input_ids, cdr3_attention_masks), dim=1)
    epitope_combined = torch.cat((epitope_input_ids, epitope_attention_masks), dim=1)

    cdr3_train_data, cdr3_test_data = train_test_split_no_shuffle(
        cdr3_combined, test_size=0.2
    )
    cdr3_labels_train_data, cdr3_labels_test_data = train_test_split_no_shuffle(
        cdr3_labels, test_size=0.2
    )
    epitope_train_data, epitope_test_data = train_test_split_no_shuffle(
        epitope_combined, test_size=0.2
    )

    cdr3_train_data = cdr3_train_data.transpose(0, 1)
    cdr3_test_data = cdr3_test_data.transpose(0, 1)
    epitope_train_data = epitope_train_data.transpose(0, 1)
    epitope_test_data = epitope_test_data.transpose(0, 1)

    cdr3_train_data = cdr3_train_data.to(device)
    cdr3_test_data = cdr3_test_data.to(device)
    epitope_train_data = epitope_train_data.to(device)
    epitope_test_data = epitope_test_data.to(device)

    # leave only 10% of the labels in the data (just to test it)
    data_amount = 1
    print("Using ", data_amount * 100, "% of the data")
    if data_amount != 1:
        unique = np.unique(cdr3_labels_train_data)
        np.random.seed(42)
        np.random.shuffle(unique)
        np.sort(unique)
        the_labels = [
            np.where(cdr3_labels_train_data == x)[0]
            for x in unique[: int(len(unique) * 0.1)]
        ]
        the_labels = np.concatenate(the_labels)

        cdr3_labels_train_data = cdr3_labels_train_data[the_labels]
        cdr3_train_data[0] = cdr3_train_data[0][the_labels]
        cdr3_train_data[1] = cdr3_train_data[1][the_labels]
    loss_func = torch.nn.TripletMarginLoss(
        margin=1.0, p=2.0, eps=1e-06, swap=False, reduction="mean"
    )

    attention_heads = [4, 8, 16]
    hidden_sizes = [32, 64, 128]
    num_hidden_layers = [2, 4, 8]
    intermediate_sizes = [32, 64, 128]
    i = 0
    epochs = 1000
    batch_size = 32

    # flatten configs
    for hidden_size in hidden_sizes:
        for num_hidden_layer in num_hidden_layers:
            for num_attention_heads in attention_heads:
                for intermediate_size in intermediate_sizes:
                    config = BertConfig(
                        hidden_size=hidden_size,
                        hidden_act="gelu",
                        initializer_range=0.02,
                        vocab_size=25,
                        hidden_dropout_prob=0.1,
                        num_attention_heads=num_attention_heads,
                        type_vocab_size=2,
                        max_position_embeddings=32,
                        num_hidden_layers=num_hidden_layer,
                        intermediate_size=intermediate_size,
                        attention_probs_dropout_prob=0.1,
                    )
                    early_stopper = EarlyStopper(
                        patience=100, min_delta=0, val_data_min_delta=0.001
                    )
                    model_save_path = f"./runs/model_hidden_size_{hidden_size}_num_hidden_layer_{num_hidden_layer}_intermediate_size_{intermediate_size}_num_attention_heads_{num_attention_heads}"
                    model = build_model(config)
                    # num_params = sum(p.numel() for p in model.parameters())
                    # print(f"Total Parameters: {num_params}")
                    #
                    # print("\nModel's architecture:")
                    # print(model)
                    # print("Training the model...")
                    print(
                        "Using config: ",
                        i,
                        "out of ",
                        len(attention_heads)
                        * len(hidden_sizes)
                        * len(num_hidden_layers)
                        * len(intermediate_sizes),
                    )
                    s_t = time.time()
                    loss_dict = train_model(
                        model,
                        epochs,
                        batch_size,
                        cdr3_train_data,
                        cdr3_labels_train_data,
                        cdr3_test_data,
                        cdr3_labels_test_data,
                        loss_func,
                        early_stopper,
                        save_every_x_epochs=10,
                        model_save_path=model_save_path,
                    )
                    print(
                        "it took ",
                        round(time.time() - s_t),
                        "seconds to train the model",
                    )
                    i += 1
