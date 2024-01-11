import time

from prettytable import PrettyTable
from svm_trainer import SVMTrainer
from svm_kernel import Kernel

def calculate(true_positive,false_positive,false_negative):
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
    f = open("/content/drive/MyDrive/SpamFilter/Results/results.txt","a")
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
    f.write("Time spent for model : " + str(round(time()-start_time,2)))
    f.write("\n")
    f.write("\n")
    f.write("\n")
    f.close()
