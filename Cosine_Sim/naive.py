
import numpy as np

def naive_cos_similarity(a, b):
    #convert to numpy arr for operations
    a,b = np.asarray(a), np.asarray(b)
    
    dot_p = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_p / (norm_a * norm_b)




if __name__=="__main__":
    # a = [1, 2, 3]   #only works with 1-D vectors of the same size, also cannot handle 0-valued vectors -> would return Nan 
    # b = [4, 5, 6]

    a = list(map(int,(input().split())))
    b = list(map(int,(input().split())))
    
    print(a)
    
    print(naive_cos_similarity(a, b))