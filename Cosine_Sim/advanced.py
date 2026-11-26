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
    scores= cosine_sim_matrix(Q, D)
    order = np.argsort(-scores, axis=1, kind='stable')
    return order[:, :k]


if __name__=='__main__':
    pass