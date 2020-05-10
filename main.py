import numpy as np

# Rescore and rank

# Scores for a particular boundary and concept
def rescore(scores, scores_shifted):
    score_arr = scores - scores_shifted
    score_arr[score_arr <= 0] = 0
    K = scores.shape[0]
    return (1/K) * np.sum(scores)
# 
# z
# n - (layer,concept)
# z + lambda * n

# fi(g(z))

# fi(g(z + lambda*n))

# Find the most easily manipulatable concept for a particular layer
def rank_concepts(scores, scores_shifted):
    """
    scores - list of arrays of scores (one list per concept)
    """

    C = len(scores)

    rescore_list = []

    for c in range(C):
        c_scores = scores[c]
        c_scores_shifted = scores_shifted[c]
        c_rescore = rescore(scores, scores_shifted)
        rescore_list.append(c_rescore)
    

    
    # Get the best concept
    best_concept = np.argmax(rescore_list)
    return best_concept

def get_best_concepts(L, scores, scores_shifted):
    # L - number of layers
    # scores - C lists
    # scores_shifted - L sets each of C lists

    best_concept = []

    for l in L:        
        bc = rank_concepts(scores, scores_shifted[l])
        best_concept.append(bc)
    
    return best_concept

