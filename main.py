import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gradient_descent(bias, lambda1, alpha, X, Y, max_iter):

    for i in range(max_iter):
        hx = X*lambda1 + bias
        print("hx (linear regression function) has value: ",hx)

        # have exapanded one line code into 3 for the understanding
        dLambda1 = (hx-Y)* X
        dLambda1 = np.divide(dLambda1, 2*(X.shape[0]))
        dLambda1 = np.sum(dLambda1)
        print("dLambda1 (weight derivative): ", dLambda1)

        dbias = np.sum(np.divide((hx-Y), 2*(X.shape[0])))
        print("dbias (bias derivative): ", dbias)

        #apply gradient descent
        lambda1 = lambda1 - alpha * dLambda1
        bias = bias - alpha * bias

        max_iter += 1

    return bias, lambda1

def linearRegression(X, Y):

    # set initial parameters for model
    bias = 0.3
    lambda1 = 2

    alpha = 0.5 # learning rate
    max_iter=50

    #TODO
    # call gredient decent to calculate intercept(=bias) and slope(lambda1)
    bias, lambda1 = gradient_descent(bias, lambda1, alpha, X, Y, max_iter)
    print ('Final bias and  lambda1 values are = ', bias, ' and ', lambda1, " respecively." )

    # plot the data and overlay the linear regression model
    yPredictions = (lambda1*X)+bias
    plt.scatter(X, Y)
    plt.plot(X,yPredictions,'k-')
    plt.show()

    
def main():
    
    # Read data into a dataframe
    df = pd.read_excel('data.xlsx')
    df = df.dropna() 

    # Convert Dataframe to a NumPy array
    X = df.values
    plt.scatter(X[:,1], X[:,0])
    plt.show()

    # Store feature and target data in separate arrays
    Y = df['Y'].values
    X = df['X'].values


    # Perform standarization on the feature data
    X = (X - np.mean(X))/np.std(X)

    linearRegression(X, Y)

main()
