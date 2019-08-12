# multiclass_svm

''' image classification using multiclass SVM '''
import numpy as np

def L_i_vectorized(x, y, W):
    ''' 
    loss function for svm - uses hinge method (
    returns zero if score at least greater than 1 + any other class score, else difference) 
    - good idea to add a regularization term to formula l*R(W)
    '''
    scores = W.dot(x)
    margins - np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i

def preprocessing(X, kind="mean"):
    if kind == "mean":
        X -= np.mean(X, axis = 0)
    elif kind == "normal":
        X /= np.std(X, axis = 0)
    else:
        print("Error: Please choose 'mean' or 'normal'.")
    return X

