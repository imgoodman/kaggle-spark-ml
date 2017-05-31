#-*- coding:utf8-*-
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint,LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
import numpy as np
import operator
import matplotlib.pyplot as plt

train_file_path="/usr/bigdata/data/houseprice/noheader_train.csv"

sc=SparkContext("local[2]","spark kaggle house price regression")

raw_data=sc.textFile(train_file_path).map(lambda line:line.split(","))

type_columns=[2,5,7,8,9,10,11,12,13,14,15,16,21,22,23,24,27,28,29,39,40,41,53,55,65,78,79]
type_columns_with_NA=[6,25,30,31,32,33,35,42,57,58,60,63,64,72,73,74]

number_columns=[1,4,17,18,19,20,34,36,37,38,43,44,45,46,47,48,49,50,51,52,54,56,61,62,66,67,68,69,70,71,75,76,77]
number_columns_with_NA=[3,26,59]
number_columns_with_many_zeros=[26,34,36,37,38,44,45,62,66,67,68,69,70,71,75]

saleprice_column=80

def getMapOfColumn(idx):
    return raw_data.map(lambda fields:fields[idx]).distinct().zipWithIndex().collectAsMap()

def get_type_maps():
    type_maps={}
    for i in type_columns:
        type_maps[i]=getMapOfColumn(i)
    """
    包含未知的类型
    """
    for i in type_columns_with_NA:
        type_maps[i]=getMapOfColumn(i)
    return type_maps

type_maps=get_type_maps()
#print type_maps

def get_type_cnt(maps):
    return sum([len(maps[i]) for i in maps])
    
type_cnt=get_type_cnt(type_maps)
number_cnt=len(number_columns)
total=type_cnt+number_cnt

total_dt=len(type_columns)+len(type_columns_with_NA)+len(number_columns)

#print total

def extract_features(fields):
    features=np.zeros(total)
    step=0
    for i in type_columns:
        features[step+ int(type_maps[i][fields[i]]) ]=1.0
        step=step+len(type_maps[i])
    """
    包含未知的类型
    """
    for i in type_columns_with_NA:
        features[step+int(type_maps[i][fields[i]])]=1.0
        step=step+len(type_maps[i])
    for i in number_columns:
        features[step]=float(fields[i])
        step=step+1
    return features

def extract_features_dt(fields):
    features=np.zeros(total_dt)
    step=0
    for i in type_columns:
        features[step]=float(type_maps[i][fields[i]])
        step=step+1
    """
    包含未知的类型
    """
    for i in type_columns_with_NA:
        features[step]=float(type_maps[i][fields[i]])
        step=step+1
    for i in number_columns:
        features[step]=float(fields[i])
        step=step+1
    return features

data=raw_data.map(lambda fields: LabeledPoint(float(fields[saleprice_column]),extract_features(fields)))
data_dt=raw_data.map(lambda fields: LabeledPoint(float(fields[saleprice_column]),extract_features_dt(fields)))
print(data.take(10))
#print data_dt.first()


def squared_error(actual, pred):
    return (actual - pred)**2

def abs_error(actual, pred):
    return np.abs(actual - pred)

def squared_log_error(actual, pred):
    return (np.log(actual+1) - np.log(pred+1))**2

def actual_pred_error(actual_vs_pred):
    mse=actual_vs_pred.map(lambda actual, pred : squared_error(actual, pred)).mean()
    mae=actual_vs_pred.map(lambda actual, pred : abs_error(actual, pred)).mean()
    rmsle=np.sqrt(actual_vs_pred.map(lambda actual, pred : squared_log_error(actual, pred)).mean())
    print( "mse is: %f" % mse)
    print( "mae is: %f" % mae)
    print( "rmsle is: %f" % rmsle)
    return (mse, mae, rmsle)

def predict_lr():
    data_with_idx=data.zipWithIndex().map(lambda k,v : (v, k))
    test=data_with_idx.sample(False, 0.2, 42)
    train=data_with_idx.subtractByKey(test)
    train_data=train.map(lambda idx,p : p)
    test_data=test.map(lambda idx,p : p)
    print( "train data size: %d" % train_data.count())
    print( "test data size: %d" % test_data.count())
    # number of iterations
    params=[1, 5, 10, 20, 50, 100]
    #step in each iteration
    steps=[0.01, 0.025, 0.05, 0.1, 1.0]
    # regularization
    regTypes=['l1', 'l2']
    regParam=[0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
    intercepts=[True, False]
    metrics=[evaluate(train_data, test_data, param, 0.01, 0.0, 'l2', False) for param in params]
    print( params)
    print( metrics)

def evaluate(train, test, iterations, step, regParam, regType, intercept):
    lrModel=LinearRegressionWithSGD.train(train, iterations, step,regParam=regParam, regType=regType, intercept=intercept)
    # weights of lr model
    # lrModel.weights
    actual_vs_pred=test.map(lambda p: (p.label, lrModel.predict(p.features)))
    #print actual_vs_pred.take(10)
    actual_pred_error(actual_vs_pred)
    
#predict_lr()

def predict_dt():
    data_with_idx=data_dt.zipWithIndex().map(lambda k,v : (v,k))
    test=data_with_idx.sample(False, 0.2, 42)
    train=data_with_idx.subtractByKey(test)
    test_data=test.map(lambda idx,p:p)
    train_data=train.map(lambda idx,p:p)
    maxDepths=[1,2,3,4,5,10,20]
    maxBins=[2,4,8,16,32,64,100]
    m={}
    for maxDepth in maxDepths:
        for maxBin in maxBins:
            metrics=evaluate_dt(train_data, test_data, maxDepth, maxBin)
            print( "metrics in maxDepth: %d; maxBins: %d" % (maxDepth, maxBin))
            print( metrics)
            m["maxDepth:%d;maxBins:%d" % (maxDepth, maxBin)]=metrics[2]
    mSort=sorted(m.iteritems(), key=operator.itemgetter(1), reverse=True)
    print( mSort)

def evaluate_dt(train, test, maxDepth, maxBins):
    dtModel=DecisionTree.trainRegressor(train, {}, impurity='variance', maxDepth=maxDepth, maxBins=maxBins)
    preds=dtModel.predict(test.map(lambda p: p.features))
    actual=test.map(lambda p: p.label)
    actual_vs_pred=actual.zip(preds)
    #print actual_vs_pred.take(10)
    #print "decision tree depth: %d" % dtModel.depth()
    #print "decision tree number of nodes: %d" % dtModel.numNodes()
    return actual_pred_error(actual_vs_pred)
#predict_dt()
