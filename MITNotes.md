# Fati's notes from [MIT Intro to Machine Learning:Transformers Chapter 9](https://introml.mit.edu/notes/transformers.html)

## 9.1 Transformers Overview
- auto-regressive: generates sequences by predicting each token sequentially, conditioned on previously generated tokens
- architecture processes inputs by applying multiple identical building blocks stacked in layers.
    - each block consists of an attention layer and a feed-forward network. 
- transformers have capacity for parallel processing.

## 9.2 Embedding and Representations
- two key components to any ML system: 1. representation of the data and 2. actual modelling to perform task
- tokenization: determining how to parse individual words
    - ex: 3 individual tokens can be (eval + ua + tion)

## 9.3
- Attention mechanisms efficiently process global info by selectively focusing on the most relavent parts of the input. 
- attention: mechanism that enables models to identify and prioritize contextually relevant tokens. 
### 9.3.1 Query Vector
- query vector: probes other tokens including itself to determine relevance
- query vector is defined by a learnable query weight matrix $W_q$ and the input token $x_i$.
    - $ q_i = W_q^T x_i $ 
    - for a sequence of n tokens we generate n distinct query vectors.
### 9.3.2 Key Vectors
- tokens used to answer queries about their relavance. When evaluating token $x_3$ its query vector $q_3$ is compared to each token's key vector $k_j$ to determing the attention weight.
    - $k_i = W_k^T x_i$
- The attention mechanism calculates similarity using the dot product which efficiently measures vector similarity:
    - $a_i = softmax(\frac{[q_i^Tk_1,q_i^Tk_2,...,q_i^Tk_n]}{\sqrt{d_k}}^T \exists \real^{1 x n})$
    - vector $a_i$ quantifies how much attention token $q_i$ should pay to each token in the sequence - normalized. 
### 9.3.3 Attention Output
- vector $z_i$ represents token $x_i$ enriched embedding, incorporating context from accross the sequence weighted by learned attention
    - $z_i = \sum^n_j=1 a_{i,j,}v_j \exist \real^{d_k}$

## 9.4 Self-attention layer
- self-attention: mechanism where the keys, values, and queries are all generated from the same input.
### 9.4.1 A single sef-attention head
- layer takes in n tokens, each having feature dimension d.
- all tokens can be collectively written as $X \exists \real^{n x d}$, where the i-th row of X stroes the ith token: $x_i \exists \real^{1 x d}$.
- self-attention computes (via projection matrices), query, keys, and value and overall we will have n of each. 
    - all of these vectors live in the same dimension in practice deonted via $d_k$
- self-attention output is calculated as a weigthed sum
    - $ z_i = \sum^n_{j=1} a_{i,j}v_j \exists \real^d_k $
    - the Q,K, V matrices stack in ro-wise manner. and softmax operation is applied in row-wise manner. 
    - $Attention(Q,K,V) = softmax_{row}(\frac{QK^T}{\sqrt{d_k}}V)
### 9.4.2 Multi-head Self-attention
- the model uses multiple attention heads, eahc with its own independently learned set of {Q,K,V} matrices. 
    - this allows each head to attend to different parts of the input tokens and to model different types of semantic relationships. 
    - the different "perspectives" are concatenated and projected to produce a richer more expressive representation of the input. 
- denote the number of heads as H. 
    - ${Q,K,V}^h = XW_{q,k,v}^h$
    - output: $Z^h = Attention(Q^h,K^h,V^h) \exists \real^{n x d_k}$
- $ MultiHead(X) = Concat(Z^1,...,Z^H)W^O$
    - $W^O$ is a final linear projection matrix.

## 9.5 Transformers Architecture Details
### 9.5.1 Positional Embeddings
- transformers incorporate positional embeddings - additional information that encodes the position of each token in the sequence.
### 9.5.2 Causal Self-attention
- masks can be applied to limit which tokens are used in the attention coputation. 
    - used to prevent the attention  mechanism from "looking ahead"
