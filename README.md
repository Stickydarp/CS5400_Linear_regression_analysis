# Machine Learning Report
**By:** Kevin Mccole  

## Goal
Implement linear and ridge regression and test them on a dataset, then evaluate each model for its competency.

## Workflow
1. **Read in the data** from Excel file  
2. **Shuffle and divide** the data into training, validation, and test sets  
3. **Train the models**  
4. **Test the models** against the test data  
5. **Compare** results against scikit-learn  

## Linear Regression Implementation
```python
def linearRegression(X,Y):
    A = np.insert(X,0,1,axis=1)#adding bias
    AT = np.transpose(A)
    #W=(AT*A)^-1 *AT*Y
    w = np.linalg.pinv((AT @ A))@(AT @ Y)
    return w
```
### Linear Regression equation:  (AT*A)^-1 * (AT*Y)
### I used Pinv to calculate the inverse because I was getting very large or small numbers which is indicative of an unstable calculation 

## Ridge Regression Implimentation
```python
def ridgeRegression(X, Y, L):
    A = np.insert(X, 0, 1, axis=1)  # adding bias
    AT = A.T
    n, p = A.shape
    I = np.eye(p)
    I[0,0] = 0
    w = np.linalg.pinv(AT @ A + L * I)@(AT @ Y)
    return w
```
### Ridge regression equation: (A*AT + L*I)^-1 * (AT * Y)
## Ridge Fitting Implimentation
```python
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
```


