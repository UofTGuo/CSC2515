import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
#np.random.seed(8069)

BATCHES = 50

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch    


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    n = X.shape[0]
    gradient = 2*((np.dot(np.dot(X.T, X), w))-(np.dot(X.T,y)))/(n)
    return gradient


def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    #3.5
    K = 500 
    grad_sum = 0
    for i in range(K):
        X_sample, y_sample = batch_sampler.get_batch()
        batch_grad = lin_reg_gradient(X_sample, y_sample, w)
        grad_sum = grad_sum +batch_grad
    sample_grad = grad_sum/K
    print(sample_grad)
    
    true_grad = lin_reg_gradient(X, y, w)
    print(true_grad)
    
    #cosine similarity
    cos_similarity = cosine_similarity(sample_grad,true_grad)
    print(cos_similarity)
    
    #squared distance metric    
    squared_dist = 0
    for i in range(len(sample_grad)):
        squared_dist = squared_dist + ((sample_grad[i] - true_grad[i])**2)
    print(squared_dist)

    #3.6
    K = 500
    log_m = list()
    log_variances = list()
    for m in range(1, 401):
        log_m.append(np.log(m))
        batch_sampler = BatchSampler(X,y,m)
        batch_grad = list()
        for k in range(K):
            X_sample, y_sample = batch_sampler.get_batch()
            batch_grad.append(lin_reg_gradient(X_sample,y_sample,w))
        w_param_grad = list()
        for i in batch_grad:
            #change to other values from 0 to 12
            w_param_grad.append(i[0])
        log_variances.append(np.log(np.var(w_param_grad))) 
    plt.plot(log_m,log_variances,".")
    plt.title("log of m vs log of sample variance(w_0)")
    plt.xlabel("log of m")
    plt.ylabel("log of sample variance of mini-batch gradient estimate")      
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
