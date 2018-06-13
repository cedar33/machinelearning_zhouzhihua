import math
import numpy as np

def sigmoid_fun(z=0.0):
    return 1/(1+math.exp(-z))


def liner_fun(x, w, b=0.0):
    x = np.array(x)
    w = np.array(w)
    return np.dot(w, x)+b


def learning_algrithm(x, y, iteration=1000, lr=1):
    x = np.array(x, dtype='float32')
    y = np.array(y, dtype='float32')
    w = [0]*len(x[0])
    b = [0.0]
    param_beta = np.array(w+b)
    param_xsh = np.concatenate([x, [[1.0]]*len(x)], axis=1)
    likehood = likehood_fun(param_beta, param_xsh, x, y)
    print('likehood', likehood)
    newtton = de1_fun(y, param_xsh, param_beta)/de2_fun(param_xsh, param_beta)
    # print('newtton--------------', newtton)
    for i in range(iteration):
        param_beta = param_beta - lr*newtton
        newtton = de1_fun(y, param_xsh, param_beta)/de2_fun(param_xsh, param_beta)
        likehood_new = likehood_fun(param_beta, param_xsh, x, y)
        # print('likehood---------------', likehood_new-likehood)
        likehood = likehood_new
    return param_beta


def likehood_fun(param_beta, param_xsh, x, y=0.0):
    x = np.array(x,dtype='float32')
    likehood = 0.0
    for i in range(len(param_xsh)):
        # print(np.dot(param_beta, param_xsh[i]))
        tmp = -1*y[i]*np.dot(param_beta, param_xsh[i]) + math.log(1 + math.exp(np.dot(param_beta, param_xsh[i])))
        likehood += tmp
    return likehood


def de1_fun(y, param_xsh, param_beta):
    de1 =  np.zeros_like(param_xsh[0], dtype='float32')
    for i in range(len(param_xsh)):
        p1 = p1_fun(param_beta, param_xsh[i])
        tmp = param_xsh[i]*(y[i]-p1)
        de1 += tmp
    return -de1


def de2_fun(param_xsh, param_beta):
    de2 = 0.0
    for i in range(len(param_xsh)):
        p1 = p1_fun(param_beta, param_xsh[i])
        tmp = np.dot(param_xsh[i], param_xsh[i])*p1*(1-p1)
        de2 += tmp
    return de2
    

def p1_fun(param_beta, param_xshi):
    tmp =  math.exp(np.dot(param_beta, param_xshi))
    return tmp/(1 + tmp)



data_list = list()
with open(r'D:\\周志华机器学习\data\watermellon.txt', 'r', encoding='utf-8') as f:
    while 1:
        line = f.readline()
        if line:
            data_list.append(str(line).split(','))
        else:
            break

data_nparr = np.array(data_list)
x_data = data_nparr[:,0:2]
y_data = data_nparr[:,2]
learning_algrithm(x_data, y_data, iteration=500)

