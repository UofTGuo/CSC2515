from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
#np.random.seed(8069)

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        max_x = max(X[:,i])
        min_x = min(X[:,i])
        max_y = max(y)
        min_y = min(y)
        plt.plot(X[:,i],y, ".")
        plt.axis([min_x-((max_x - min_x)*0.1),max_x+((max_x - min_x)*0.1),
                  min_y-((max_y - min_y)*0.1),max_y+((max_y - min_y) *0.1)])
        plt.xlabel(features[i])
        plt.ylabel("Median value(in $1000's)")
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    #left part (XTX)
    left = np.dot(X.T,X)
    #right part (XTy)
    right = np.dot(X.T,Y)
    w = np.linalg.solve(left,right)    
    return w

def MSE(X,y,w):
    return np.sum((np.dot(X,w)-y)**2)/(X.shape[0])

def MAE(X,y,w):
    return np.sum(np.abs((np.dot(X,w)-y)))/(X.shape[0])
    
def R_squared(X,y,w):    
    y_mean = np.add(np.zeros(y.shape[0]),np.mean(y))
    numerator = np.dot((y - np.dot(X, w)).T,(y - np.dot(X, w)))
    denominator = np.dot((y - y_mean).T,(y - y_mean))
    R_squared = 1 - numerator/denominator
    return R_squared


def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)

    #adding error term
    b = list()
    for i in range(X.shape[0]):
        a = [1] + X[i].tolist()
        b.append(a)
    X = np.array(b)    
    
    #TODO: Split data into train and test
    new_index = np.random.choice(X.shape[0],X.shape[0],replace=False)
    training_index = new_index[:int(X.shape[0]*0.8),]
    test_index = new_index[int(X.shape[0]*0.8):,]
    X_training = X[training_index]
    X_test = X[test_index]
    y_training = y[training_index]
    y_test = y[test_index]
    
    # Fit regression model
    w = fit_regression(X_training, y_training)
    print(w)

    # Compute error measurement metrics
    print("Test set MSE is:", MSE(X_test,y_test,w))
    print("Test set MAE is:", MAE(X_test,y_test,w))
    print("Test set R squared is:", R_squared(X_test, y_test, w))
    


if __name__ == "__main__":
    main()

