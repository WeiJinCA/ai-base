import spacy
import numpy as np
import pandas as pd

text1 = "我喜欢的水果是橙子和苹果"
text2 = "相比苹果，我更喜欢国产的华为"

# 1 Embedding the text using the small Chinese model
# Load the small Chinese model
nlp = spacy.load("zh_core_web_sm")
doc = nlp(text1)

#Dimentions of the embedding
emd_dim = 10

dics = {}
for token in doc:
    dics[token.text] = token.vector[:emd_dim]
#print(dics)

X = pd.DataFrame(dics)
#print(X.T) # 8 rows × 10 columns

# 2 Initialize the Q, K, V matrices
d_k = 6 #Dimension of the key, will be used in the softmax function
Wq = np.random.rand(emd_dim, d_k) #10x6
Wk = np.random.rand(emd_dim, d_k) #10x6
Wv = np.random.rand(emd_dim, d_k) #10x6
#print(Wq)

#Calculate the Q, K, V matrices
#Q = np.dot(X.T, Wq) #8x6
Q = X.T @ Wq
print(Q)
K = X.T @ Wk
V = X.T @ Wv

# Scale the Q, K, V matrices
df_QK = Q@K.T / np.sqrt(d_k)

# Claculate the softmax
for i in range(len(df_QK)):
    exp_v = np.exp(df_QK.iloc[i])
    softmax = exp_v / np.sum(exp_v)
    df_QK.iloc[i] = softmax
print(df_QK)

#Calculate the V matrix
attention = df_QK @ V