from proj1_helpers import standardize
import numpy as np
def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

def calc_MI(X,Y, bins, bins_Y):
    c_XY = np.histogram2d(X, Y, bins)[0]
    c_X = np.histogram(X, bins)[0]
    c_Y = np.histogram(Y, bins_Y)[0]
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI

def top_features(tx_cat, y_cat, max_features):
    bins = 50
    top_features = []
    other_features = []
    
    
    for i in range(len(tx_cat)):
    
        # Standardize Data, for iterative methods
        tx_std, mean_x, std_x = standardize(tx_cat[i])
        y = y_cat[i]

        vecMI = np.zeros(np.shape(tx_std)[1])

        for j in range(np.shape(tx_std)[1]):
            vecMI[j] = calc_MI(tx_std[:, j], y, bins, 2)

        top_features.append(np.argsort(-vecMI)[:max_features])
        other_features.append(np.argsort(-vecMI)[max_features:])
        
        print('Top features for category {}: {}'.format(i,top_features[i]))
        
    return top_features, other_features
