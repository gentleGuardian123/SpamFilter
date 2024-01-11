import time
import pandas as pd
import cvxopt.solvers

from prettytable import PrettyTable
from svm_trainer import SVMTrainer
from svm_kernel import Kernel
from root_path import root_path

def calculate(true_positive,false_positive,false_negative, true_negative):
    result = {}
    result['precision'] = true_positive / (true_positive + false_positive)
    result['recall'] = true_positive / (true_positive + false_negative)
    return result

def confusion_matrix(true_positive,false_positive,false_negative,true_negative):
    matrix = PrettyTable([' ', 'Ham' , 'Spam'])
    matrix.add_row(['Ham', true_positive , false_positive])
    matrix.add_row(['Spam', false_negative , true_negative])
    return matrix , calculate(true_positive,false_positive,false_negative,true_negative)

def implement(X_train, Y_train, X_test, Y_test, params, type):
    ham_spam = 0
    spam_spam = 0
    ham_ham = 0
    spam_ham = 0
    if(type=="polykernel"):
        dimension = params['dimension']
        offset = params['offset']
        trainer = SVMTrainer(Kernel.polykernel(dimension,offset),0.1)
        predictor = trainer.train(X_train,Y_train)
    elif(type=="linear"):
        trainer = SVMTrainer(Kernel.linear(),0.1)
        predictor = trainer.train(X_train,Y_train)
    for i in range(X_test.shape[0]):
        ans = predictor.predict(X_test[i])
        print(ans)
        if(ans==-1 and Y_test[i]==-1):
            spam_spam+=1
        elif(ans==1 and Y_test[i]==-1):
            spam_ham+=1
        elif(ans==1 and Y_test[i]==1):
            ham_ham+=1
        elif(ans==-1 and Y_test[i]==1):
            ham_spam+=1
    return confusion_matrix(ham_ham,ham_spam,spam_ham,spam_spam)

def write_to_file(matrix,result,parameters,type,start_time):
    f = open(root_path + "results.txt","a")
    if(type=="polykernel"):
        f.write("Polykernel model parameters")
        f.write("\n")
        f.write("Dimension : " + str(parameters['dimension']))
        f.write("\n")
        f.write("Offset : " + str(parameters['offset']))
    elif(type=="linear"):
        f.write("Linear model")
    f.write("\n")
    f.write(matrix.get_string())
    f.write("\n")
    f.write("Precision : " + str(round(result['precision'],2)))
    f.write("\n")
    f.write("Recall : " + str(round(result['recall'],2)))
    f.write("\n")
    f.write("Time spent for model : " + str(round(time.time()-start_time,2)))
    f.write("\n")
    f.write("\n")
    f.write("\n")
    f.close()

def train_and_test():
    global_start_time = time.time()
    cvxopt.solvers.options['show_progress'] = False

    # df2 = pd.read_csv(root_path + "frequency.csv", header=0)
    df2 = pd.read_csv(root_path + "test_frequency.csv", header=0)

    input_output = df2.values
    X = input_output[:,:-1]
    Y = input_output[:,-1:]
    train = int(X.shape[0] * 70 / 100)

    X_train = X[:train,:]
    Y_train = Y[:train,:]
    X_test = X[train:,:]
    Y_test = Y[train:,:]

    # f = open(root_path + "results.txt","w+")
    f = open(root_path + "test_results.txt","w+")
    f.close()
    k = 0
    type = {}
    parameters = {}
    type['1'] = "polykernel"
    type['2'] = "linear"

    for i in range(2,4):
        for j in range(0,10):
            start_time = time.time()
            parameters['dimension'] = i
            parameters['offset'] = j
            matrix, result = implement(X_train,Y_train,X_test,Y_test,parameters,str(type['1']))
            write_to_file(matrix,result,parameters,type,start_time)
            k += 1
            print("Finished files: " + str(k))

    start_time = time.time()
    matrix , result = implement(X_train,Y_train,X_test,Y_test,parameters,str(type['2']))
    write_to_file(matrix,result,parameters,type,start_time)
    k += 1
    print("Finished files: " + str(k))

    # f = open(root_path + "results.txt","a")
    f = open(root_path + "test_results.txt","a")
    f.write("Time spent for entire code : " + str(round(time.time()-global_start_time,2)))
    f.close()