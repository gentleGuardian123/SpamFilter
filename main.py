import pandas as pd
import cvxopt.solvers

from time import time
from support_vector_machine import *

if __name__ == '__main__':

    global_start_time = time()
    cvxopt.solvers.options['show_progress'] = False

    df1 = pd.read_csv('/content/drive/MyDrive/SpamFilter/Results/wordslist.csv')
    df2 = pd.read_csv('/content/drive/MyDrive/SpamFilter/Results/frequency.csv',header=0)

    # input_output = df2.as_matrix(columns=None)
    input_output = df2.values
    X = input_output[:,:-1]
    Y = input_output[:,-1:]
    total = X.shape[0]
    train = int(X.shape[0] * 70 / 100)

    X_train = X[:train,:]
    Y_train = Y[:train,:]
    X_test = X[train:,:]
    Y_test = Y[train:,:]

    f = open("/content/drive/MyDrive/SpamFilter/Results/results.txt","w+")
    f.close()
    k = 0
    type = {}
    parameters = {}
    type['1'] = "polykernel"
    type['2'] = "linear"

    for i in range(2,4):
        for j in range(0,10):
            start_time = time()
            parameters['dimension'] = i
            parameters['offset'] = j
            matrix , result = implementSVM(X_train,Y_train,X_test,Y_test,parameters,str(type['1']))
            write_to_file(matrix,result,parameters,type,start_time)
            k += 1
            print("Finished files: " + str(k))

    start_time = time()
    matrix , result = implementSVM(X_train,Y_train,X_test,Y_test,parameters,str(type['2']))
    write_to_file(matrix,result,parameters,type,start_time)
    k += 1
    print("Finished files: " + str(k))

    f = open("/content/drive/MyDrive/SpamFilter/Results/results.txt","a")
    f.write("Time spent for entire code : " + str(round(time()-global_start_time,2)))
    f.close()