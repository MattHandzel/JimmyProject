from transformers import BertForMaskedLM, BertConfig
import pickle
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from early_stopper import EarlyStopper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config, verbose=1):
    model = BertForMaskedLM(config=config)  # change this line and flag what we're using
    model.to(device)
    return model


def get_anchor_positive_index_negative_index(labels):
    random_label = np.random.choice(np.unique(labels))
    random_labels_we_want = labels[labels == random_label]
    negative_index = np.random.choice(np.where(labels != random_label)[0])
    indexs = (np.where(labels == random_label))[0]
    random_index_0 = None
    try:
        random_index_0 = np.random.choice(indexs)
    except ValueError:
        print(random_index_0, random_label, indexs)

    indexs = np.delete(indexs, np.where(indexs == random_index_0))
    if len(indexs) == 0:
        return random_index_0, random_index_0, negative_index
    random_index_1 = np.random.choice(indexs)

    return random_index_0, random_index_1, negative_index


def get_anchors_positives_and_negatives(labels, label_positives_index_map):
    # For every label, get a positive and negative then return the indexes
    anchors = []
    positives = []
    negatives = []
    for i, anchor in enumerate(labels):
        anchors.append(i)
        positive = np.random.choice(label_positives_index_map[anchor])
        positives.append(positive)
        other_keys = list(label_positives_index_map.keys())
        other_keys.remove(anchor)
        negative = np.random.choice(
            label_positives_index_map[np.random.choice(other_keys)]
        )
        # print(
        #     "anchor label",
        #     anchor,
        #     "pos label",
        #     labels[positive],
        #     "neg label",
        #     labels[negative],
        #     "ai",
        #     i,
        #     "pi",
        #     positive,
        #     "ni",
        #     negative,
        # )
        negatives.append(negative)

    return anchors, positives, negatives


def train_model(
    model,
    num_epochs,
    batch_size,
    x_train,
    x_train_labels,
    x_test,
    x_test_labels,
    loss_func,
    early_stopper: EarlyStopper = None,
    save_every_x_epochs=-1,
    model_save_path="",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-5)
    batch_size = 128
    train_data_size = x_train.shape[1]
    test_data_size = x_test.shape[1]
    losses = []
    test_losses = []
    label_test_positives_index_map = {}
    for label in x_test_labels:
        # Compute labels positive index mpa
        if label not in label_test_positives_index_map:
            label_test_positives_index_map[label] = np.where(x_test_labels == label)[0]
    label_positives_index_map = {}
    for label in x_train_labels:
        # Compute labels positive index mpa
        if label not in label_positives_index_map:
            label_positives_index_map[label] = np.where(x_train_labels == label)[0]
    for epoch in range(num_epochs):
        model.train()

        anchor_indexs = []
        positive_indexs = []
        negative_indexs = []
        (
            anchor_indexs,
            positive_indexs,
            negative_indexs,
        ) = get_anchors_positives_and_negatives(
            x_train_labels, label_positives_index_map
        )
        train_loss = 0
        for batch in range(0, train_data_size, batch_size):
            end_batch = batch + batch_size
            if end_batch > train_data_size:
                end_batch = train_data_size
            anchor_output = model(
                x_train[0][anchor_indexs[batch : batch + batch_size]],
                x_train[1][anchor_indexs[batch : batch + batch_size]],
            )
            positive_output = model(
                x_train[0][positive_indexs[batch : batch + batch_size]],
                x_train[1][positive_indexs[batch : batch + batch_size]],
            )
            negative_output = model(
                x_train[0][negative_indexs[batch : batch + batch_size]],
                x_train[1][negative_indexs[batch : batch + batch_size]],
            )

            loss = loss_func(anchor_output[0], positive_output[0], negative_output[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().numpy() * batch_size

        # model evaluation
        #
        model.eval()

        (
            anchor_indexs,
            positive_indexs,
            negative_indexs,
        ) = get_anchors_positives_and_negatives(
            x_test_labels, label_test_positives_index_map
        )
        test_loss = 0
        for batch in range(0, test_data_size, batch_size):
            end_batch = batch + batch_size
            if end_batch > test_data_size:
                end_batch = test_data_size
            anchor_output = model(
                x_test[0][anchor_indexs[batch : batch + batch_size]],
                x_test[1][anchor_indexs[batch : batch + batch_size]],
            )
            positive_output = model(
                x_test[0][positive_indexs[batch : batch + batch_size]],
                x_test[1][positive_indexs[batch : batch + batch_size]],
            )
            negative_output = model(
                x_test[0][negative_indexs[batch : batch + batch_size]],
                x_test[1][negative_indexs[batch : batch + batch_size]],
            )
            loss = loss_func(anchor_output[0], positive_output[0], negative_output[0])
            test_loss += loss.detach().numpy() * batch_size

        train_loss /= train_data_size
        losses.append(train_loss)

        test_loss /= test_data_size
        test_losses.append(test_loss)
        if save_every_x_epochs > 0:
            if epoch % save_every_x_epochs == 0:
                # print(model_save_path)
                # if not os.path.exists(model_save_path):
                #     os.makedirs(model_save_path)
                model.save_pretrained(model_save_path + f"/epoch_{epoch}")
        if early_stopper is not None:
            print(f"Test loss {test_loss}", early_stopper.min_validation_loss)
            if early_stopper.early_stop(test_loss):
                print("Early stopping")
                break
        print(
            f"Epoch: {epoch} | Train loss: {train_loss} | Test loss {test_loss}"  # {summed_test_loss / (eval_every_x_steps / output_every_x_steps)}"
        )
    model.save_pretrained(model_save_path + "/final_model")

    plt.plot(losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()
    plt.savefig(model_save_path + "/loss.png")
    with open(model_save_path + "train_loss.pickle", "wb") as file:
        pickle.dump(losses, file)
    with open(model_save_path + "test_loss.pickle", "wb") as file:
        pickle.dump(test_losses, file)

    print("Array has been successfully saved using pickle.")
    return {"train_loss": losses, "test_loss": test_losses}
