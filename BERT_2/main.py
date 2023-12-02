import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import time
import pickle
from transformers import *
from dataloader import *
from early_stopper import EarlyStopper
import pickle
import argparse
from model import *

if __name__ == "__main__":
    hidden_size = 16
    lr = 1e-4
    num_hidden_layer = 2
    num_attention_heads = 2
    intermediate_size = 16
    # parse the arguments (which will be the hyperapameters, if there is nothing there, default to the hyperparameters up top)
    epochs = 100
    batch_size = 32
    parser = argparse.ArgumentParser(
        description="Arguments (hidden_size, lr, num_hidden_layer, num_attention_heads, intermediate_size)\nEx.: python main.py --lr 1e-4 --num_hidden_layer 2 --num_attention_heads 2 --intermediate_size 16 --epochs 100 --batch_size 32"
    )

    parser.add_argument(
        "--hidden_size", type=int, default=hidden_size, help="hidden size of the model"
    )
    parser.add_argument("--lr", type=float, default=lr, help="learning rate")
    parser.add_argument(
        "--num_hidden_layer",
        type=int,
        default=num_hidden_layer,
        help="number of hidden layers",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=num_attention_heads,
        help="number of attention heads",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=intermediate_size,
        help="intermediate size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=epochs,
        help="number of epochs to train the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=batch_size,
        help="batch size to train the model",
    )

    args = parser.parse_args()

    hidden_size = args.hidden_size
    lr = args.lr
    num_hidden_layer = args.num_hidden_layer
    num_attention_heads = args.num_attention_heads
    intermediate_size = args.intermediate_size
    epochs = args.epochs
    batch_size = args.batch_size

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

    cdr3_input_ids, cdr3_attention_masks = pad_sentences(
        cdr3_sentences, max_length, tokenizer
    )
    epitope_input_ids, epitope_attention_masks = pad_sentences(
        epitope_sentences, max_length, tokenizer
    )

    cdr3_combined = torch.cat((cdr3_input_ids, cdr3_attention_masks), dim=1)
    epitope_combined = torch.cat((epitope_input_ids, epitope_attention_masks), dim=1)

    data_amount = 1
    print("Using ", data_amount * 100, "% of the data")

    shuffle_indexs = np.random.permutation(len(cdr3_labels))

    cdr3_labels = cdr3_labels[shuffle_indexs]
    cdr3_combined = cdr3_combined[shuffle_indexs]
    epitope_combined = epitope_combined[shuffle_indexs]

    cdr3_labels = cdr3_labels[: round(len(cdr3_labels) * data_amount)]
    cdr3_combined = cdr3_combined[: round(len(cdr3_combined) * data_amount)]
    epitope_combined = epitope_combined[: round(len(epitope_combined) * data_amount)]

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

    loss_func = torch.nn.TripletMarginLoss(
        margin=1.0, p=2.0, eps=1e-06, swap=False, reduction="mean"
    )

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
    early_stopper = EarlyStopper(patience=100, min_delta=0, val_data_min_delta=0.05)
    model_save_path = f"./runs/model_hidden_size_{hidden_size}_num_hidden_layer_{num_hidden_layer}_intermediate_size_{intermediate_size}_num_attention_heads_{num_attention_heads}"
    model = build_model(config)
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
        save_every_x_epochs=1,
        model_save_path=model_save_path,
        learning_rate=lr,
    )
    print(
        "it took ",
        round(time.time() - s_t),
        "seconds to train the model",
    )
