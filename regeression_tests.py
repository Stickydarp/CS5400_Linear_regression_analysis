import numpy as np
import pandas as pd


def linearRegression(X,Y):
    A = np.insert(X,0,1,axis=1)#adding bias
    AT = np.transpose(A)
    #W=(AT*A)^-1 *AT*Y
    
    w = (np.linalg.inv(AT@A))@(AT@Y) 
    return w

def ridgeRegression(X,Y,L): 
    A = np.insert(X,0,1,axis=1)#adding bias
    AT = np.transpose(A)
    n,p=A.shape
    I = np.eye(p)
    IL=I*L

    #W=(AT*A)^-1 *AT*Y

    w=(np.linalg.inv(AT@A +IL)@(AT@Y))
    return w

def findBestFit(TX,TY,VX,VY,LL,UL,Increment):
    VXBiased= np.insert(VX,0,1,axis=1)#adding bias

    tests=int((UL-LL)/Increment)
    bestLambda=0
    bestWeight=None
    bestLoss=float("inf")
    
    for test in range(0,tests):
        L=LL+(Increment*test)
        w=ridgeRegression(TX,TY,L)
        y_pred = VXBiased @ w
        loss = np.sum((VY - y_pred) ** 2) + L * np.sum(w[1:] ** 2)
        if loss < bestLoss:
            bestLoss= loss
            bestWeight=w
            bestLambda=L
        
    print(f"Found that the best Lambda for the validation set is :{bestLambda}\nThis lambda produced a loss of: {bestLoss}")
    
    return bestWeight,bestLambda






if __name__ == "__main__":
    
    ###(1)take in data
    df = pd.read_excel( "ENB2012_data.xlsx" , sheet_name ="sheet1")
    #converting to numpy matrix
    matrix = df.to_numpy(dtype=float)


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
    
    optimizedRidgeWeights,L=findBestFit(trainingX,trainingY,validationX,validationY,0,0.0005,0.0000001)
    print(optimizedRidgeWeights)



    ###(4)validate using validation data
    
    


    
    ###(5) test on test data and give accuracy
        
    testXBiased= np.insert(testX,0,1,axis=1)#adding bias
    ridge_y_pred = testXBiased @ optimizedRidgeWeights
    ridge_loss = np.sum((testY - ridge_y_pred) ** 2) + L * np.sum(optimizedRidgeWeights[1:] ** 2)*(1/numColumns+1)
    
    
    linear_y_pred = testXBiased @ linearWeights
    linear_loss = np.sum((testY - linear_y_pred) ** 2)*(1/numColumns+1) 
    
    print(f"linear loss is {linear_loss} and ridge loss is {ridge_loss}")

        
