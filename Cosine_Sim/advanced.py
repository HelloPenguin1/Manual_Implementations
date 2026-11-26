import numpy as np

def normalize_rows(X):
    norms = np.linalg.norm(X, axis = 1, keepdims=True)
    #handle division by zero edge case
    safe = np.where(norms>0, norms, 1.0)
    
    return X/safe

def cosine_sim_matrix(Q, D):
    '''Calculates cosine similarity between every row of Q and every row of D'''
    return normalize_rows(Q) @ normalize_rows(D).T

def top_k(Q, D, k):
    '''sorts scores in descending order (highest score first), and trims based on top k'''
    scores= cosine_sim_matrix(Q, D)
    order = np.argsort(-scores, axis=1, kind='stable')
    return order[:, :k]



'''
This advanced version of cosine similarity is more realistic..
ASSUMPTIONS:
- Q and D are 2D matrices where each row represents one embeddding vector
- Both have same num of cols - required for norm and matrix calcs


HOW IS IT MORE ROBUST:
- replaces zero norms by 1
- processes multiple embeddings simultaenousl

The naive implementation calculates cosine similarity between two individual vectors. The advanced implementation extends this to matrices of embeddings using row-wise L2 normalization and matrix multiplication.

It is more robust because it handles zero-norm vectors through safe division, supports batch comparisons, and provides stable top-K document ranking.

However, it still assumes compatible embedding dimensions and vector spaces, and requires additional input validation for production use

'''
