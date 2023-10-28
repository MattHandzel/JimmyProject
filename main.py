# %%
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# %%
positives = pd.read_csv("./positive.csv")
negatives = pd.read_csv("./negative.csv")

all_amino_acids = []

def add_to_all_amino_acids(acid_sequence:str):
  for acid in acid_sequence:
    if acid not in all_amino_acids:
      all_amino_acids.append(acid)

# Figure out all of the amino acids that are in the sequences and make a label encoder based off of that
positives.stack().reset_index(drop=True).apply(lambda x: add_to_all_amino_acids(x))
sorted(all_amino_acids) # sort it for funsies
amino_acid_label_encoder = LabelEncoder()
amino_acid_label_encoder.fit(all_amino_acids)
all_amino_acids = amino_acid_label_encoder.transform(all_amino_acids)

def convert_protien_sequence_to_feature_map(protien_sequence):
  '''
  This function takes in a protien sequence and uses one-hot encoding to encode each protein in a matrix
  '''
  return protien_sequence.apply(lambda x: tf.one_hot(amino_acid_label_encoder.transform(list(x)), len(all_amino_acids)))

data = convert_protien_sequence_to_feature_map(positives["cdr3"])

# %%
# do some goofy pca stuff
summed_data = np.array(list(data.apply(lambda x: np.sum(x,0)).values))[:100]
pca = PCA(n_components=3)
pca.fit(summed_data)
summed_data = pca.transform(summed_data)

# clustering
cluster = AgglomerativeClustering( n_clusters=5,affinity='euclidean', linkage='ward')
n_clusters = cluster.n_clusters
labels = cluster.fit_predict(summed_data)

fig = plt.figure()
ax = plt.axes(projection='3d')

# Plot and color the data based off of its cluster number

for n in range(n_clusters):
  ax.scatter3D(summed_data[labels == n, 0], summed_data[labels == n, 1], summed_data[labels == n, 2], color=np.random.uniform(0, 1, 3), marker='o')
plt.show()

# %%
pca.explained_variance_ratio_

# %%
np.array(list(summed_data)).shape

# %%



