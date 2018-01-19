# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:38:56 2017

@author: Jason

A Logistic Regression model to calculate from a random data set of blood samples and multiple features.
The model is built manually and the accuracy is computed
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import numpy.random as rand
import scipy.optimize as opt
import random
from numpy import genfromtxt
from sklearn import cross_validation
import time

class LogisticRegression(object):
    
    def __init__(self,path = "C:\\Users\\Jason\\Documents\\GitHub\\Logistic-Regression-python\\realdata1.csv"):
        
        dfRD = pd.read_csv(path,error_bad_lines = False,names='a')
        self.df = dfRD['a'].str.split(",",expand=True)
        del self.df[25]
        self.df.columns =['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane','class']

        
    def convertToNumeric(self):
        
        #cleaning the file
        df = self.df
        df = df.convert_objects(convert_numeric=True)

        df.replace('\t"?','?',inplace = True)
        for ecol in df.columns:
            if df[ecol].dtypes == 'float64':
 
                df[ecol].fillna(df[ecol].mean(skipna = True),inplace=True)

            else:
                df[ecol].replace('?',df[ecol].mode()[0],inplace=True)
                df[ecol].fillna(df[ecol].mode()[0],inplace=True)
        

        df.replace('\t"43',43,inplace = True)
        df.replace('\t"6200',6200,inplace = True)
        df.replace('\t"8400',8400,inplace = True)
        df.replace('notckd\t','notckd',inplace = True)
        df.replace('ckd\t','ckd',inplace = True)
        df.replace('\t"no','no',inplace = True)
        df.replace('\t"yes','no',inplace = True)
        df.replace(' yes','no',inplace = True)
        df.replace('','no',inplace = True)
        df = df[df.astype(str).ne('None').all(1)]

        li = ['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
        
        for ecol in li:
            df[ecol] = (df[ecol] - df[ecol].min())/(df[ecol].max()-df[ecol].min())
       
        
        target = df[['class']]
        df.drop(['class'],1,inplace = True)

        numeric = pd.get_dummies(df,columns = ['sg','al','su','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane'])
        
        for i in range(target.shape[0]):
            if target.iloc[i][0] == 'ckd':
                target.iloc[i][0] = 1
            else:
                target.iloc[i][0] = 0
        
        target = target.apply(pd.to_numeric)
        numeric = numeric.convert_objects(convert_numeric=True)
        target = target.astype(float)
        
        return numeric.as_matrix(),target.ix[:,0]

    def sigmoid(self,h):
    	return 1/(1+ np.exp(-h))
    	 
     	
    def predict(self,w,p):			

    	prediction =self.sigmoid(np.dot(p.T,w.T))
    	if prediction > 0.5:
    			prediction = 1
    	else:
    			prediction = 0
    	return prediction  


    def run(self):
        Xdf,Ydf = self.convertToNumeric()
        Ydf.reset_index(drop=True,inplace=True)
        

        '''replace automaic shuffle with manual shuffle, problem is the data is heavily skewed so fmeasure goes to infinity'''
        #TrainingSize = int (Xdf.shape[0]*0.8)
        #XTrain = Xdf[:TrainingSize,:]
        #YTrain = Ydf[:TrainingSize]
        #XTest = Xdf[TrainingSize:,:]
        #YTest = Ydf[TrainingSize:]
        #YTest.reset_index(drop=True,inplace=True)
        
        
        
        XTrain,XTest,YTrain,YTest = cross_validation.train_test_split(Xdf,Ydf,test_size=0.2)
        YTest.reset_index(drop=True,inplace=True)
        YTrain.reset_index(drop=True,inplace=True)
        
        
        XTrain = np.insert(XTrain, len(XTrain[1]), 1, axis=1)     
        XTest = np.insert(XTest, len(XTest[1]), 1, axis=1)
        w = np.zeros((1,len(XTrain[1])))
        for i in range(w.shape[1]):
        		w[0][i] = np.random.randint(100)/1000
                
        
        self.LamdaList = np.arange(-2.0,4.0,0.2)
        self.fmeasureTestList =[]
        self.fmeasureTrainList =[]
        
        for value in self.LamdaList:
            X,y,w,lamda,alpha = XTrain,YTrain,w,value,0.001
    
    
            w_initialize = np.random.rand(len(X[1]))
            alpha = 0.01
    
            w = w_initialize
            iteration_number = 30
            for i in range(iteration_number):
            	wNew = w
            	for ele in range(len(X[1])):
            		wNew[ele] = w[ele] - alpha*np.dot((1/(1+np.exp(-np.matmul(w, X.transpose()))) - y),X[:,ele])-lamda/len(X)*w[ele]
            	w = wNew
    
            right = 0
            wrong = 0
            FP = 0
            FN = 0 
            TP=  0
            TN = 0
            for i in range(XTest.shape[0]):
                p = XTest[i,:,None]
                predicta = self.predict(w,p)
    
                if predicta == YTest[i]:
                    right+=1
                else:
                    wrong+=1
                if (predicta == YTest[i]):
                    if(predicta == 0):
                        TN+=1
                    else:
                        TP+=1
                else:
                    if(predicta == 1):
                        FP+=1
                    else:
                        FN+=1
            

            if (TP+FP) == 0:
                pre = 0
            else:
                pre = (TP/(TP+FP))
            if (TP+FN) == 0:
                rec = 0
            else:
                rec = (TP/(TP+FN))
            
            self.fmeasureTestList.append((2*pre*rec)/(pre+rec))
            
            FP = 0
            FN = 0 
            TP=  0
            TN = 0
            for i in range(XTrain.shape[0]):
                p = XTrain[i,:,None]
                predicta = self.predict(w,p)
    
                if predicta == YTrain[i]:
                    right+=1
                else:
                    wrong+=1
                if (predicta == YTrain[i]):
                    if(predicta == 0):
                        TN+=1
                    else:
                        TP+=1
                else:
                    if(predicta == 1):
                        FP+=1
                    else:
                        FN+=1
            

            if (TP+FP) == 0:
                pre = 0
            else:
                pre = (TP/(TP+FP))
            if (TP+FN) == 0:
                rec = 0
            else:
                rec = (TP/(TP+FN))

            
            self.fmeasureTrainList.append((2*pre*rec)/(pre+rec))

        
    def plotGraph(self):
         plt.plot(self.LamdaList,self.fmeasureTestList,label="Test")  
         plt.plot(self.LamdaList,self.fmeasureTrainList,label="Train")  
         plt.xlabel("Lambda")
         plt.ylabel("fmeasure")
         plt.title("Logistic Regression Problem b")
         plt.legend()
        
def main():
    start = time.time()
    a = LogisticRegression("C:\\Users\\Jason\\Documents\\GitHub\\Logistic-Regression-python\\realdata1.csv")
    a.run()
    a.plotGraph()      
    end = time.time()
    print("Time measure: %.2f sec"  %(end-start))

    
if __name__ == "__main__":
    main()  