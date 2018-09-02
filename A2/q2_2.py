'''
Question 2.2 Skeleton Code
Here you should implement and evaluate the Conditional Gaussian classifier.
'''

from __future__ import division
import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class
    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    #version 1
    means = []
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        # Compute mean of class i
        means.append(np.mean(i_digits, axis=0))
    means = np.reshape(means,(10,64))
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class
    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        for j in range(64):
            for k in range(64):
                i_indx = []
                for indx in range(len(train_data)):
                    if (train_labels[indx] ==i):
                        i_indx.append(indx)
                N = len(i_indx)
                X_j = [train_data[indx][j] for indx in i_indx]
                X_k = [train_data[indx][k] for indx in i_indx]
                cov_j_k = (1/N)*sum([(X_j[n]-means[i][j])*(X_k[n]-means[i][k]) for n in range(N)])
                covariances[i][j][k] = cov_j_k
    for label in range(10):
        covariances[label] = covariances[label] + 0.01*np.identity(64)
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    cov_diagonals = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        cov_diagonals.append(np.reshape(cov_diag, (8, 8)))
    all_concat = np.concatenate(np.log(cov_diagonals),1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()
  
def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)
    Should return an n x 10 numpy array 
    '''
    result = []
    for digit in digits:
        log_lik = []
        for k in range(10):
            exp_part = np.exp((-1 / 2) * np.linalg.multi_dot([np.transpose(digit - means[k]), np.linalg.inv(covariances[k] + 0.01 * np.identity(64)),digit - means[k]]))
            log_lik.append(np.log(((2*np.pi)**(-64/2)) * (np.linalg.det(covariances[k] + 0.01*np.identity(64)) ** (-1/2)) * exp_part))
        result.append(log_lik)
    return result
    
def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:
        log p(y|x, mu, Sigma)
    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    result = []
    prob_y = 1/10
    generative_log_lik = generative_likelihood(digits, means, covariances)
    for gen in generative_log_lik:
        log_prob_x = np.log(sum(np.exp(gen_feature_d) for gen_feature_d in gen)*prob_y)
        digit_cond_lik = []
        for gen_feature_d in gen:
            digit_cond_lik.append(gen_feature_d + np.log(prob_y) - log_prob_x)
        result.append(digit_cond_lik)
    return result
    
    #generative = generative_likelihood(digits, means, covariances)
    #px = np.sum(np.exp(generative - np.log(10)), axis=1).reshape((digits.shape[0], )) # 700 x 1
    #px = np.full((10, digits.shape[0]), px).T
    #log_px = np.log(px)
    #results = (generative - np.log(10)) - log_px
    #return results    

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels
        AVG( log p(y_i|x_i, mu, Sigma) )
    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    #cond_likelihoods = conditional_likelihood(digits, means, covariances)
    #total_likelihoods = 0
    #for n_indx, cond_like in enumerate(cond_likelihoods):
    #    total_likelihoods += cond_like[int(labels[n_indx])]
    #return total_likelihoods / len(digits)
    
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    temp = np.zeros(cond_likelihood.shape)
    for i in range(temp.shape[0]):
        j = int(labels[i])
        temp[i, j] = 1
    result = np.sum(temp * cond_likelihood) / temp.shape[0]
    return result    

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    #cond_likelihood = conditional_likelihood(digits, means, covariances)
    #result = []
    #for indx, cond_l in enumerate(cond_likelihood):
    #    result.append(max([(k, prob) for k, prob in enumerate(cond_l)], key=lambda x: x[1])[0])
    #return result
    
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    classes = np.arange(10)
    indices = np.argmax(cond_likelihood, axis=1)
    return classes[indices]    

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    #Part (1)
    plot_cov_diagonal(covariances)

    # Evaluation
    
    #Part(2)
    avg_cond_like_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    avg_cond_like_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    
    print("when k is 1, the training accuracy is",classification_accuracy(knn, 1, train_data, train_labels))
    print("when k is 1, the test accuracy is",classification_accuracy(knn, 1, test_data, test_labels))    
    
    #Part(3)
    classes_train = classify_data(train_data, means, covariances)
    classes_test = classify_data(test_data, means, covariances)
    class_accuracy_train = sum(1 for indx, k in enumerate(classes_train) if train_labels[indx] == k)/len(train_data)
    class_accuracy_test = sum(1 for indx, k in enumerate(classes_test) if test_labels[indx] == k)/len(test_data)
    print("Average conditional likelihood ->", "\nTrain: ", avg_cond_like_train, "\nTest: ", avg_cond_like_test)
    print("Classification accuracy->", "\nTrain: ", class_accuracy_train, "\nTest: ", class_accuracy_test)
    

if __name__ == '__main__':
    main()