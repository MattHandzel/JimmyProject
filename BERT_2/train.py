import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import matplotlib.pyplot as plt


if __name__ == "__main__":
  positives = pd.read_csv("./positive.csv")
  negatives = pd.read_csv("./negative.csv")

  amino_acids = []

  def add_to_amino_acids(a_sequence: str):
      for acid in a_sequence:
          if acid not in amino_acids:
              amino_acids.append(acid)

  positives.stack().reset_index(drop=True).apply(add_to_amino_acids)

  amino_acids.sort()

  amino_acid_label_encoder = LabelEncoder()
  amino_acid_label_encoder.fit(amino_acids)

  all_amino_acids = amino_acid_label_encoder.transform(amino_acids)

  def feature_map(p_sequence):
      return [tf.one_hot(amino_acid_label_encoder.transform(list(x)), len(all_amino_acids)) for x in p_sequence]

  data_cd3r = feature_map(positives["cdr3"])
  data_epitope = feature_map(positives["antigen.epitope"])


  def convert_to_space_separated_string(series):
      return ' '.join(series)

  tokenizer = BertTokenizer(vocab_file="./amino_acid_vocab.txt")

  def construct_sentences(dataframe):
      cdr3_sentences = dataframe["cdr3"]
      epitope_sentences = dataframe["antigen.epitope"]
      return cdr3_sentences, epitope_sentences

  def pad_sentences(sentences, max_length):
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
      
  max_length = 32

  positives = pd.read_csv("./positive.csv")

  cdr3_sequence_epitope_encoder = LabelEncoder()
  cdr3_sequence_epitope_encoder.fit(positives["antigen.epitope"])

  cdr3_labels = cdr3_sequence_epitope_encoder.transform(positives["antigen.epitope"])
  for column in positives.columns:
      positives[column] = positives[column].apply(convert_to_space_separated_string)

  cdr3_sentences, epitope_sentences = construct_sentences(positives)

  shuffle_data(cdr3_sentences, 42)
  shuffle_data(epitope_sentences, 42)
  shuffle_data(cdr3_labels, 42)

  cdr3_input_ids, cdr3_attention_masks = pad_sentences(cdr3_sentences, max_length)
  epitope_input_ids, epitope_attention_masks = pad_sentences(epitope_sentences, max_length)

  cdr3_combined = torch.cat((cdr3_input_ids, cdr3_attention_masks), dim=1)
  epitope_combined = torch.cat((epitope_input_ids, epitope_attention_masks), dim=1)


  cdr3_train_data, cdr3_test_data = train_test_split_no_shuffle(cdr3_combined, test_size=0.2)
  cdr3_labels_train_data, cdr3_labels_test_data = train_test_split_no_shuffle(cdr3_labels, test_size=0.2)
  epitope_train_data, epitope_test_data = train_test_split_no_shuffle(epitope_combined, test_size=0.2)



  cdr3_train_data = cdr3_train_data.transpose(0,1)
  cdr3_test_data = cdr3_test_data.transpose(0,1)
  epitope_train_data = epitope_train_data.transpose(0,1)
  epitope_test_data = epitope_test_data.transpose(0,1)

  # leave only 10% of the labels in the data (just to test it)
  unique = np.unique(cdr3_labels_train_data)
  np.random.seed(42)
  np.random.shuffle(unique) 
  np.sort(unique)
  the_labels = [np.where(cdr3_labels_train_data == x)[0] for x in unique[:int(len(unique) * 0.1)]]
  the_labels = np.concatenate(the_labels)

  cdr3_labels_train_data = cdr3_labels_train_data[the_labels]
  cdr3_train_data[0] = cdr3_train_data[0][the_labels]
  cdr3_train_data[1] = cdr3_train_data[1][the_labels]
  # cdr3_labels_test_data = cdr3_labels_test_data[:int(len(cdr3_labels_test_data) * 0.1)]

  from transformers import *

  config = BertConfig.from_json_file("./bert_config.json")

  model = BertForMaskedLM(config=config) # change this line and flag what we're using

  device = "cuda" if torch.cuda.is_available() else "cpu"

  cdr3_train_data = cdr3_train_data.to(device)
  cdr3_test_data = cdr3_test_data.to(device)
  epitope_train_data = epitope_train_data.to(device)
  epitope_test_data = epitope_test_data.to(device)

  model.to(device)

  # print("Model's state_dict:")
  # for param_tensor in model.state_dict():
  #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

  print("\nModel's parameters:")
  num_params = sum(p.numel() for p in model.parameters())
  print(f'Total Parameters: {num_params}')

  print("\nModel's parameters (only those that require gradients):")
  num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f'Total Trainable Parameters: {num_trainable_params}')

  print("\nModel's architecture:")
  print(model)
# Testing the loss function...
loss_func = torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, reduction='mean')
from transformers import BertTokenizer, BertForPreTraining, BertForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, BertConfig

# Load the BERT tokenizer
config = BertConfig(
  hidden_size= 64,
  hidden_act= "gelu",
  initializer_range= 0.02,
  vocab_size= 25,
  hidden_dropout_prob= 0.1,
  num_attention_heads= 8,
  type_vocab_size= 2,
  max_position_embeddings= 64,
  num_hidden_layers= 3,
  intermediate_size= 64,
  attention_probs_dropout_prob= 0.1
)

model = BertForMaskedLM(config=config).to() # change this line and flag what we're using
%%time
steps = int(1e6)
X_train = cdr3_train_data
optimizer = torch.optim.Adam(model.parameters(), lr=10e-5)
batch_size = 128
output_every_x_steps = 100
eval_every_x_steps = 100

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

losses = []
test_losses = []
summed_loss = 0
summed_test_loss = 0
for step in range(steps):
    ### Training
    model.train() # train mode is on by default after construction
    
    # print(loss)

    # 3. Zero grad optimizer
    optimizer.zero_grad()
    loss = run_model_on_input_and_get_triplet_loss(model, batch_size, cdr3_train_data, cdr3_labels_train_data, loss_func)
    # 4. Loss backward
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()
    summed_loss += loss.detach().numpy()

    # ### Testing
    # if (step % (eval_every_x_steps) == 0) and step != 0:
    #     model.eval() # put the model in evaluation mode for testing (inference)
    #     with torch.inference_mode():
    #         test_loss = run_model_on_input_and_get_triplet_loss(model, batch_size, cdr3_test_data, cdr3_labels_test_data, loss_func)
    #     summed_test_loss += test_loss.detach().numpy()
    #     test_losses.append(test_loss.detach().numpy())
        
    losses.append(loss.detach().numpy())
    if step % output_every_x_steps == 0 and step != 0:
        print(f"Step: {step} | Train loss: {summed_loss / output_every_x_steps} | Test loss {summed_test_loss / (eval_every_x_steps)}")
        summed_loss = 0
        summed_test_loss = 0
plt.plot(losses)
plt.plot(test_losses)