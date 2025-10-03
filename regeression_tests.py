import numpy as np
import pandas as pd


def linearRegression(X,Y):
    A = np.insert(X,0,1,axis=1)#adding bias
    AT = np.transpose(A)
    #W=(AT*A)^-1 *AT*Y
    
    w = np.linalg.pinv((AT @ A))@(AT @ Y) 
    return w

def ridgeRegression(X, Y, L):
    A = np.insert(X, 0, 1, axis=1)  # adding bias
    AT = A.T
    n, p = A.shape
    I = np.eye(p)
    I[0,0] = 0
    w = np.linalg.pinv(AT @ A + L * I)@(AT @ Y)
    return w



def findBestFit(TX, TY, VX, VY,lambdas):
    VXBiased = np.insert(VX, 0, 1, axis=1)
    bestLambda, bestWeight, bestLoss = 0, None, float("inf")

    #ridge_loss = (np.sum((testY - ridge_y_pred) ** 2) + L * np.sum(optimizedRidgeWeights[1:] ** 2)) / len(testY)
    for L in lambdas: 
        w = ridgeRegression(TX, TY, L)
        y_pred = VXBiased @ w
        loss = (np.sum((VY - y_pred) ** 2 ) + L * np.sum(w[1:] ** 2)) / len(VY)
        if loss < bestLoss:
            bestLoss, bestWeight, bestLambda = loss, w, L

    print(f"Best λ = {bestLambda}, validation loss = {bestLoss}")
    return bestWeight, bestLambda



if __name__ == "__main__":
    
    ###(1)take in data
    df = pd.read_excel( "ENB2012_data.xlsx" , sheet_name ="sheet1")
    #converting to numpy matrix
    matrix = df.to_numpy(dtype=float)
    np.random.seed(4444)
    np.random.shuffle(matrix)

    #print(matrix)

    ###(2)split up data
    #70% training
    #10% validation
    #20% test
    
    numRows,numColumns = matrix.shape
   
    trainingBound=(numRows//10)*7
    validationBound=(numRows//10)*8
    
    ## X
    trainingX=matrix[:trainingBound,:-2]
    validationX=matrix[trainingBound:validationBound,:-2]
    testX=matrix[validationBound:,:-2]
    ## Y
    trainingY=matrix[:trainingBound,-2:]
    validationY=matrix[trainingBound:validationBound,-2:]
    testY=matrix[validationBound:,-2:]

    
        
    ###(3)train models
    linearWeights=linearRegression(trainingX,trainingY)
    print(linearWeights)
    
    #ridgeWeights=ridgeRegression(trainingX,trainingY,1)
    #print(ridgeWeights)
    
    lambdas = np.logspace(-4, 2, 100)
    optimizedRidgeWeights,L=findBestFit(trainingX,trainingY,validationX,validationY,lambdas)
    print(optimizedRidgeWeights)
 


    
    ###(4) test on test data and give accuracy
        
    testXBiased = np.insert(testX, 0, 1, axis=1)
    ridge_y_pred = testXBiased @ optimizedRidgeWeights
    #ridge_loss = (np.sum((testY - ridge_y_pred) ** 2) + L * np.sum(optimizedRidgeWeights[1:] ** 2)) / len(testY)

    linear_y_pred = testXBiased @ linearWeights
    #linear_loss = np.sum((testY - linear_y_pred) ** 2) / len(testY)


    ridge_loss = np.sum((testY - ridge_y_pred) ** 2) / testY.size
    linear_loss = np.sum((testY - linear_y_pred) ** 2) / testY.size


    print(f"Linear MSE = {linear_loss:.4f}, Ridge MSE = {ridge_loss:.4f}")
    
        ### (5) Compare with scikit-learn
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error

    # sklearn Linear Regression
    sk_linear = LinearRegression(fit_intercept=True)
    sk_linear.fit(trainingX, trainingY)
    sk_linear_preds = sk_linear.predict(testX)
    sk_linear_loss = mean_squared_error(testY, sk_linear_preds)

    # sklearn Ridge Regression (with same best lambda found)
    sk_ridge = Ridge(alpha=L, fit_intercept=True)
    sk_ridge.fit(trainingX, trainingY)
    sk_ridge_preds = sk_ridge.predict(testX)
    sk_ridge_loss = mean_squared_error(testY, sk_ridge_preds)

    print("\n--- Scikit-learn Comparison ---")
    print(f"My Linear MSE     = {linear_loss:.4f}")
    print(f"sklearn Linear MSE = {sk_linear_loss:.4f}")
    print(f"My Ridge MSE      = {ridge_loss:.4f} (λ={L:.6f})")
    print(f"sklearn Ridge MSE  = {sk_ridge_loss:.4f} (λ={L:.6f})")

    


