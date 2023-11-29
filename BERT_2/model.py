from transformers import BertForMaskedLM, BertConfig
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(config ,verbose = 1):
    model = BertForMaskedLM(config=config) # change this line and flag what we're using
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
def run_model_on_input_and_get_triplet_loss(model, batch_size, cdr3_train_data, cdr3_labels_train_data, loss_func):
    anchor_indexs = []
    positive_indexs = []
    negative_indexs = []

    for i in range(batch_size):
        anchor_index, positive_index, negative_index = get_anchor_positive_index_negative_index(cdr3_labels_train_data)
        anchor_indexs.append(anchor_index), positive_indexs.append(positive_index), negative_indexs.append(negative_index)
    # print(anchor_index, positive_index)
    # print(cdr3_train_data[0][anchor_index], cdr3_train_data[1][anchor_index])
    anchor_output = model(cdr3_train_data[0][anchor_indexs],cdr3_train_data[1][anchor_indexs])
    positive_output = model(cdr3_train_data[0][positive_indexs],cdr3_train_data[1][positive_indexs])
    negative_output = model(cdr3_train_data[0][negative_indexs],cdr3_train_data[1][negative_indexs])

    # Select anchor, positive, negative

    # 2. Calculate loss
    loss = loss_func(anchor_output[0], positive_output[0], negative_output[0])
    return loss

def train_model(model, num_epochs, batch_size,  x_train, x_train_labels, x_test, x_test_labels, loss_func):
    steps = int(10000)
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-5)
    batch_size = 128
    eval_every_x_steps = 100
    output_every_x_steps = 100

    losses = []
    test_losses = []
    summed_loss = 0
    summed_test_loss = 0
    for step in range(steps):
        ### Training
        model.train()

        optimizer.zero_grad()
        loss = run_model_on_input_and_get_triplet_loss(model, batch_size, x_train, x_train_labels, loss_func)

        loss.backward()

        optimizer.step()
        summed_loss += loss.detach().numpy()

        ### Testing
        if (step % (eval_every_x_steps) == 0) and step != 0:
            model.eval() # put the model in evaluation mode for testing (inference)
            with torch.inference_mode():
                test_loss = run_model_on_input_and_get_triplet_loss(model, batch_size, x_test, x_test_labels, loss_func)
            summed_test_loss += test_loss.detach().numpy()
            test_losses.append(test_loss.detach().numpy())
        
        losses.append(loss.detach().numpy())
        if step % output_every_x_steps == 0 and step != 0:
            print(f"Step: {step} | Train loss: {summed_loss / output_every_x_steps} | Test loss {summed_test_loss / (eval_every_x_steps / output_every_x_steps)}")
            summed_loss = 0
            summed_test_loss = 0
    
    return {"train_loss": losses, "test_loss": test_losses}