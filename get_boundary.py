import numpy as np
from sklearn import svm

# By default, scores wi
def convert_scores_to_binary_labels(scores,
                                    decision_value):
    
    labels = np.int32(scores >= decision_value)
    return labels.astype(np.bool)

def train_boundary(data,
                   scores,
                   decision_value=None,
                   chosen_num_or_ratio=None,
                   boundary_type='linear',
                   **kwargs):


    if(boundary_type != 'linear'):
        raise NotImplementedError(f'Unsupported boundary type `{boundary_type}`!')

    len_data = data.shape[0]    

    if(decision_value is not None):
        decision_value = 0
        labels = convert_scores_to_binary_labels(scores, decision_value)
    elif(chosen_num_or_ratio is not None):
        num_select = chosen_num_or_ratio
        if(num_select <= 1):
            num_select *= len_data
        
        num_0_1 = num_select // 2        

        # Ascending order
        sorted_idx = np.argsort(scores)        

        data = data[sorted_idx]
        scores = scores[sorted_idx]

        l0 = [i for i in range(num_0_1)]
        l1 = [i for i in range(len_data - num_0_1, len_data)]

        labels = np.zeros(2 * num_0_1).astype(np.int32)
        labels[l1] = 1
        labels = labels.astype(np.bool)

        data = data[l0+l1]
        scores = scores[l0+l1]    

    # TODO - select data from the whole
    # For now, just do classification
    train_linear_boundary(data, labels, **kwargs)


# Boundaries will be in 
def train_linear_boundary(data,
                          labels,                          
                          val_split_ratio=0.7):

    """
    data - Points in latent space.
     
    """
    
    len_data = data.shape[0]
    num_selected = chosen_num_or_ratio
    
    # Shuffle the data
    indices = np.random.shuffle(len_data)

    train_indices = [i for i in range(val_split_ratio * len_data)]
    val_indices = [i for i in range(len_data - val_split_ratio * len_data, len_data)]

    train_data, train_labels = data[train_indices], labels[train_indices]
    val_data, val_labels = data[val_indices], labels[val_indices]

    # Train the boundary!
    clf = svm.SVC(kernel='linear')
    clf.fit(train_data, train_labels)

    # Return the direction
    dim = data.shape[1]
    direction = clf.co/ef_.reshape(1, dim).astype(np.float32)
    return direction / np.linalg.norm(direction)
    

