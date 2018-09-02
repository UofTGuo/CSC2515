'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
import heapq
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

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
        #digit = None
        #return digit
        
        dist=self.l2_distance(test_point)
        queue=[]

        for n,item in enumerate(dist):
            queue.append((item,self.train_labels[n]))
        kSmallest=heapq.nsmallest(k,queue,key=lambda x: x[0])

        #find the kth smallest dist and count for the most frequently counted one
        #if tie happens,then pull out the one with least dist sum
        res = 'Dummy'
        hashmap = {'Dummy':[0,float('inf')]};
        for n,item in enumerate(kSmallest):
            if item[1] not in hashmap:
                hashmap[item[1]]=[1,item[0]];
            else:
                hashmap[item[1]][0]+=1;
                hashmap[item[1]][1] += item[0];
            if hashmap[res][0] < hashmap[item[1]][0]:
                res=item[1];
                continue
            if hashmap[res][0] == hashmap[item[1]][0]:
                if hashmap[res][1] > hashmap[item[1]][1]:
                    res=item[1]
        return res        
    

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    
    res=['Dummy',float('-inf')]
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        accuracy=k_fold(knn.train_data, knn.train_labels, k);
        if accuracy >= res[1]:
            res=[k,accuracy];
    return res;
    

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''

    res=np.zeros(eval_data.shape[0]);
    for i in range(eval_data.shape[0]):
        res[i] = knn.query_knn(eval_data[i],k)==eval_labels[i];
    return res.mean();  

def k_fold(data,label,k,nFold=10):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           K in the number of folds
    '''
    ## TODO
    pieceLen = data.shape[0] // nFold;
    ran=np.arange(data.shape[0]);
    accuracies=np.zeros(nFold);
    for i in range(nFold):
        train_piece=np.concatenate((ran[0:i*pieceLen],ran[(i+1)*pieceLen:-1]));
        #print (train_piece)
        data_train=data[train_piece];
        label_train=label[train_piece];
        data_test=data[i*pieceLen:(i+1)*pieceLen];
        label_test = label[i * pieceLen:(i + 1) * pieceLen];
        knn = KNearestNeighbor(data_train, label_train);
        accuracies[i] = classification_accuracy(knn, k, data_test, label_test);
    return accuracies.mean();

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    #predicted_label = knn.query_knn(test_data[0], 1)

    #Part One
    print ('Accuracy for train data with k=1: ',classification_accuracy(knn,1,train_data,train_labels),\
           '\nAccuracy for train data with k=2: ',classification_accuracy(knn,2,train_data,train_labels),\
           '\nAccuracy for train data with k=3: ',classification_accuracy(knn,3,train_data,train_labels),\
           '\nAccuracy for train data with k=4: ',classification_accuracy(knn,4,train_data,train_labels),\
           '\nAccuracy for train data with k=5: ',classification_accuracy(knn,5,train_data,train_labels),\
           '\nAccuracy for train data with k=6: ',classification_accuracy(knn,6,train_data,train_labels),\
           '\nAccuracy for train data with k=7: ',classification_accuracy(knn,7,train_data,train_labels),\
           '\nAccuracy for train data with k=8: ',classification_accuracy(knn,8,train_data,train_labels),\
           '\nAccuracy for train data with k=9: ',classification_accuracy(knn,9,train_data,train_labels),\
           '\nAccuracy for train data with k=10: ',classification_accuracy(knn,10,train_data,train_labels),\
           '\nAccuracy for train data with k=11: ',classification_accuracy(knn,11,train_data,train_labels),\
           '\nAccuracy for train data with k=12: ',classification_accuracy(knn,12,train_data,train_labels),\
           '\nAccuracy for train data with k=13: ',classification_accuracy(knn,13,train_data,train_labels),\
           '\nAccuracy for train data with k=14: ',classification_accuracy(knn,14,train_data,train_labels),\
           '\nAccuracy for train data with k=15: ',classification_accuracy(knn,15,train_data,train_labels), \
           
           '\nAccuracy for test data with k=1: ',classification_accuracy(knn,1,test_data,test_labels),\
           '\nAccuracy for test data with k=2: ',classification_accuracy(knn,2,test_data,test_labels),\
           '\nAccuracy for test data with k=3: ',classification_accuracy(knn,3,test_data,test_labels),\
           '\nAccuracy for test data with k=4: ',classification_accuracy(knn,4,test_data,test_labels),\
           '\nAccuracy for test data with k=5: ',classification_accuracy(knn,5,test_data,test_labels),\
           '\nAccuracy for test data with k=6: ',classification_accuracy(knn,6,test_data,test_labels),\
           '\nAccuracy for test data with k=7: ',classification_accuracy(knn,7,test_data,test_labels),\
           '\nAccuracy for test data with k=8: ',classification_accuracy(knn,8,test_data,test_labels),\
           '\nAccuracy for test data with k=9: ',classification_accuracy(knn,9,test_data,test_labels),\
           '\nAccuracy for test data with k=10: ',classification_accuracy(knn,10,test_data,test_labels),\
           '\nAccuracy for test data with k=11: ',classification_accuracy(knn,11,test_data,test_labels),\
           '\nAccuracy for test data with k=12: ',classification_accuracy(knn,12,test_data,test_labels),\
           '\nAccuracy for test data with k=13: ',classification_accuracy(knn,13,test_data,test_labels),\
           '\nAccuracy for test data with k=14: ',classification_accuracy(knn,14,test_data,test_labels),\
           '\nAccuracy for test data with k=15: ',classification_accuracy(knn,15,test_data,test_labels))

    #Part Three
    print(cross_validation(train_data, train_labels, k_range=np.arange(1,16)))
    
if __name__ == '__main__':
    main()