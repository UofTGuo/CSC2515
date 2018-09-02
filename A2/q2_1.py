'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

from __future__ import division
import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        #distances = [(i, dist) for i, dist in enumerate(self.l2_distance(test_point))]
        dist_lst = []
        tie_lst =[]
        cls_count = {}
        min_dist = {}        
        for i, dist in enumerate(self.l2_distance(test_point)):
            dist_lst.append((i,dist))        
        dist_lst.sort(key=lambda x: x[1])
        dist_lst = dist_lst[:k]
        for i, dist in dist_lst:
            if self.train_labels[i] not in cls_count:
                cls_count[self.train_labels[i]] = 0
                min_dist[self.train_labels[i]] = float("inf")
            cls_count[self.train_labels[i]] = cls_count[self.train_labels[i]] + 1
            if dist < min_dist[self.train_labels[i]]:
                min_dist[self.train_labels[i]] = dist
        for cls, counts in cls_count.items():
            if counts == max(cls_count.values()):
                tie_lst.append(cls)
        digit_label = [min(tie_lst, key=lambda cls: min_dist[cls])]
        return digit_label       

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    avg_accuracy_lst = [] 
    for i in k_range:
        fold_accuracy = []
        for train_index, test_index in KFold(10).split(train_data):
            x_train, x_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            knn = KNearestNeighbor(x_train, y_train)
            fold_accuracy.append(classification_accuracy(knn, i, x_test, y_test))
        print("when k is",i,"the average accuracy across folds is",np.mean(fold_accuracy))
        avg_accuracy_lst.append(np.mean(fold_accuracy))
    best = avg_accuracy_lst.index(max(avg_accuracy_lst))
    return best

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    accuracy = 0
    accumulator = 0
    N = eval_data.shape[0]
    for i in range(N):
        accumulator = accumulator + (knn.query_knn(eval_data[i], k)==eval_labels[i])
    accuracy = accumulator/N
    return accuracy   

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    #predicted_label = knn.query_knn(test_data[0], 1)

    # Part(1)
    print("when k is 1, the training accuracy is",classification_accuracy(knn, 1, train_data, train_labels))
    print("when k is 1, the test accuracy is",classification_accuracy(knn, 1, test_data, test_labels))
    
    # Part(2)
    print("when k is 15, the training accuracy is",classification_accuracy(knn, 15, train_data, train_labels))
    print("when k is 15, the test accuracy is",classification_accuracy(knn, 15, test_data, test_labels))
    
    # Part(3)
    cross_validation(train_data, train_labels)
    print("when k is 4, the training accuracy is",classification_accuracy(knn, 4, train_data, train_labels))
    print("when k is 4, the test accuracy is",classification_accuracy(knn, 4, test_data, test_labels))
    

if __name__ == '__main__':
    main()