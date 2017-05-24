#-*-coding:utf8-*-
import numpy as np
import operator
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import  SVMWithSGD, NaiveBayes,LogisticRegressionWithSGD

sc=SparkContext("local[2]","titanic spark app")

input_file="/usr/bigdata/spark/kaggle-spark-ml-app/kaggle-spark-ml/titanic/data/noheader_train.csv"

raw_data=sc.textFile(input_file)

#raw_records=raw_data.map(lambda line: line.split(",")).map(lambda (passengerId, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked):(pclass, sex, age, sibsp, parch, ticket, fare, cabin, embarked, survived))
#raw_records=raw_data.map(lambda line: line.split(",")).map(lambda records:(records[2],records[4], records[5],records[6], records[7], records[8], records[9], records[10], records[11], records[1]))
"""
注意
第四列是人名name
每个人名都包含了逗号
因此 这里以逗号分割
会将人名也进行了分割
幸运的是，目前发现所有的人名都包含逗号
因此 相当于分割后的列数比header列数多一列
"""
raw_records=raw_data.map(lambda line: line.split(","))
#raw_records.cache()
#print raw_records.take(10)

survived_idx=1
pclass_idx=2
sex_idx=5
age_idx=6
sibsp_idx=7
parch_idx=8
fare_idx=10
embarked_idx=12

pclass_map=raw_records.map(lambda fields:fields[pclass_idx]).distinct().zipWithIndex().collectAsMap()
print "pclass map is: "
print pclass_map

#pclass_countbyvalue=raw_records.map(lambda fields:int(fields[pclass_idx])).countByValue()
#print pclass_countbyvalue
"""
1:216
2:184
3:491
"""

sex_map=raw_records.map(lambda fields:fields[sex_idx]).distinct().zipWithIndex().collectAsMap()
print "sex map is: " 
print sex_map

#sex_countbyvalue=raw_records.map(lambda fields:fields[sex_idx]).countByValue()
#print sex_countbyvalue
"""
male:577
female:314
"""

def convert_age(age):
    if age=="":
        return 0.0
    else:
        return float(age)

ages=raw_records.map(lambda fields:convert_age(fields[age_idx]))
#print ages.take(10)
"""
年龄
可以按照小朋友 青少年 成年人 老年人
做一个类型区分
"""

sibsp_map=raw_records.map(lambda fields:fields[sibsp_idx]).distinct().zipWithIndex().collectAsMap()
print "sibsp map is: " 
print sibsp_map
#sibsp_countbyvalue=raw_records.map(lambda fields:fields[sibsp_idx]).countByValue()
#print sibsp_countbyvalue
"""
1:209
0:608
3:16
2:28
5:5
4:18
8:7
"""


parch_map=raw_records.map(lambda fields:fields[parch_idx]).distinct().zipWithIndex().collectAsMap()
print "parch map is: " 
print parch_map
#parch_countbyvalue=raw_records.map(lambda fields:fields[parch_idx]).countByValue()
#print parch_countbyvalue
"""
1:118
0:678
3:5
2:80
5:5
4:4
6:1
"""

def convert_embarked(embark):
    if embark=="":
        return "NA"
    else:
        return embark
embarked_map=raw_records.map(lambda fields:convert_embarked(fields[embarked_idx])).distinct().zipWithIndex().collectAsMap()
print "embarked map is: " 
print embarked_map
#embarked_countbyvalue=raw_records.map(lambda fields:convert_embarked(fields[embarked_idx])).countByValue()
#print embarked_countbyvalue

"""
Q:77
S:644
C:168
NA:2
"""


def extract_features(fields):
    features=[]
    # pclass
    pclass_vector=np.zeros(len(pclass_map))
    pclass_vector[pclass_map[fields[pclass_idx]]]=1.0

    # sex
    sex_vector=np.zeros(len(sex_map))
    sex_vector[sex_map[fields[sex_idx]]]=1.0

    # age 
    # todo

    # sibsp
    sibsp_vector=np.zeros(len(sibsp_map))
    sibsp_vector[sibsp_map[fields[sibsp_idx]]]=1.0

    # parch
    parch_vector=np.zeros(len(sibsp_map))
    parch_vector[parch_map[fields[parch_idx]]]=1.0

    # fere
    # to do

    # embarked
    embarked_vector=np.zeros(len(embarked_map))
    embarked_vector[embarked_map[convert_embarked(fields[embarked_idx])]]=1.0

    features=np.concatenate((pclass_vector, sex_vector, [convert_age(fields[age_idx])], sibsp_vector, parch_vector, [float(fields[fare_idx])], embarked_vector))
    return features

data=raw_records.map(lambda fields:LabeledPoint(float(fields[survived_idx]),extract_features(fields)))
#print len(data.first().features)
#print data.take(10)
data.cache()

def predict_SVMWithSGD(numIterations,step,regParam,regType):
    """
    SVMWithSGD.train(data,iterations=100, step=1.0, regParam=0.01, miniBatchFraction=1.0, initialWeights=None, regType='l2',intercept=False, validateData=True,convergenceTol=0.001)
    data: the training data, an RDD of LabeledPoint
    iterations: the number of iterations, default 100
    step: the step parameter used in SGD, default 1.0
    regParam: the regularizer parameter, default 0.01
    miniBatchFraction: fraction of data to be used for each SGD iteration, default 1.0
    initialWeights: the initial weights, default None
    regType: the type of regularizer used for training our model, allowed values ('l1':for using L1 regularization; 'l2':for using L2 regularization, default; None: for no regularization)
    intercept: boolean parameter which indicates the use or not of the augmented representation for training data (i.e. whether bias feature are activated or not, default False)
    validateData: boolean parameter which indicates if the algorithm should validate data before training, default True
    convergenceTol: a condition which decides iteration termination, default 0.001
    """
    svmModel=SVMWithSGD.train(data, iterations=numIterations,step=step, regParam=regParam, regType=regType)
    svmMetrics=data.map(lambda p:(p.label, svmModel.predict(p.features)))
    svmAccuracy=svmMetrics.filter(lambda (actual, pred) : actual==pred).count()*1.0/data.count()
    #print "SVMWithSGD model accuracy is: %f in %d iterations,step:%f;regParam:%f;regType:%s" % (svmAccuracy, numIterations,step,regParam,regType)
    return svmAccuracy
def test_SVMWithSGD():
    svmIterations=[10,20,50,100,200]
    svmSteps=[0.01, 0.05,0.1,0.2,0.5,1.0]
    svmRegParams=[0.01,0.05,0.1,0.2,0.5,1.0]
    svmRegTypes=['l1', 'l2']
    svmAccuracy={}
    for i in svmIterations:
        for step in svmSteps:
            for regParam in svmRegParams:
                for regType in svmRegTypes:
                    k="iterations:%d,step:%f;regParam:%f;regType:%s" % (i, step, regParam, regType)
                    svmAccuracy[k]=predict_SVMWithSGD(i,step,regParam,regType)
    #print svmAccuracy
    accuracySort=sorted(svmAccuracy.iteritems(),key=operator.itemgetter(1),reverse=True)
    print accuracySort
#test_SVMWithSGD()
"""
10:61.7%
20:61.6%
50:63%
100:61.7%
200:61.8%
"""



def predict_NaiveBayes(lamb):
    """
    NaiveBayes.train(data, lambda=1.0)
    data: the training data of RDD of LabeledPoint
    lambda: the smoothing parameter, default 1.0
    """
    naiveBayesModel=NaiveBayes.train(data, lamb)
    naiveBayesMetrics=data.map(lambda p: (p.label, naiveBayesModel.predict(p.features)))
    naiveBayesAccuracy=naiveBayesMetrics.filter(lambda (actual,pred):actual==pred).count()*1.0/data.count()
    return naiveBayesAccuracy

def test_NaiveBayes():
    lambdas=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    naiveBayesAccuracy={}
    for lamb in lambdas:
        naiveBayesAccuracy[lamb]=predict_NaiveBayes(lamb)
    accuracySort=sorted(naiveBayesAccuracy.iteritems(), key=operator.itemgetter(1), reverse=True)
    print accuracySort
#test_NaiveBayes()



def predict_LogisticRegressionWithSGD(iterations,step,regParam,regType):
    """
    LogisticRegressionWithLBFGS.train(data, iterations=100, initialWeights=None, regParam=0.0, regType='l2', intercept=False, corrections=10, tolerance=1e-06, validateData=True, numClasses=2)
    data: the training data, an RDD of LabeledPoint
    iterations: the number of iterations
    corrections: the number of corrections used in the LBFGS update. if a known updater is used for binary classification, it calls the ml implementation and this parameter will have no effect. default 10
    tolerance: the convergence tolerance of iterations for L-BFGS
    numClasses: the number of classes (i.e., outcomes) a label can take in Multinomial logistic regression, default 2
    """
    lrModel=LogisticRegressionWithSGD.train(data, iterations=iterations,step=step,regParam=regParam, regType=regType)
    lrMetrics=data.map(lambda p: (p.label, lrModel.predict(p.features)))
    lrAccuracy=lrMetrics.filter(lambda (actual,pred):actual==pred).count()*1.0/data.count()
    return lrAccuracy

def test_LogisticRegressionWithSGD():
    lrIterations=[10,20,50,100,200]
    lrRegParams=[0.01,0.05,0.1,0.2,0.5,1.0]
    lrRegTypes=['l1', 'l2']
    lrSteps=[0.01, 0.05,0.1,0.2,0.5,1.0]
    lrAccuracy={}
    for i in lrIterations:
        for regParam in lrRegParams:
            for regType in lrRegTypes:
                for step in lrSteps:
                    k="iterations:%d;step:%f;regParam:%f;regType:%s" % (i,step, regParam, regType)
                    lrAccuracy[k]=predict_LogisticRegressionWithSGD(i,step,regParam,regType)
    accuracySort=sorted(lrAccuracy.iteritems(), key=operator.itemgetter(1),reverse=True)
    print accuracySort
test_LogisticRegressionWithSGD()
