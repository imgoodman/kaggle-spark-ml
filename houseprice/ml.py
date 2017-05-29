from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
import numpy as np

train_file_path="/usr/bigdata/data/houseprice/noheader_train.csv"

sc=SparkContext("local[2]","spark house price app")

raw_data=sc.textFile(train_file_path).map(lambda line: line.split(","))
raw_data.cache()

type_columns=[1,2,5,7,8,9,10,11,12,13,14,15,16,17,18,21,22,23,24,27,28,29,39,40,41,47,48,49,50,51,52,53,54,55,56,61,65,76,77,78,79]
type_columns_with_na=[6,25,30,31,32,33,35,42,57,58,60,63,64,72,73,74]
number_columns=[4,34,36,37,38,43,44,45,46,62,66,67,68,69,70,71,75]
number_columns_with_na=[3,26]
saleprice_column=80
"""
others to be determined

T,19: YearBuilt
U,20: YearRemoAdd
BH,59: GaragYrBlt
"""

def getMapOfColumn(idx):
    return raw_data.map(lambda fields:fields[idx]).distinct().zipWithIndex().collectAsMap()

def getTypeMaps():
    type_maps={}
    for t in type_columns:
        type_maps[t]=getMapOfColumn(t)
    return type_maps

type_maps=getTypeMaps()   

def show_type_maps():
    for k,v in type_maps.items():
        print "type values of column %d is:" % k
        print v
#show_type_maps()

def total_type_columns():
    cnt=0    
    for k in type_maps:
        cnt+=len(type_maps[k])
    return cnt
type_cnt=total_type_columns()
number_cnt=len(number_columns)
total_cnt=type_cnt+number_cnt
#print type_cnt


def extract_features(fields):
    step=0
    features=np.zeros(total_cnt)
    for t_idx in type_columns:
        features[step+int(type_maps[t_idx][fields[t_idx]])]=1.0
        step=step+len(type_maps[t_idx])
    for n_idx in number_columns:
        features[step]=float(fields[n_idx])
        step=step+1
    return features

data=raw_data.map(lambda fields:LabeledPoint(float(fields[saleprice_column]),extract_features(fields)))

#first_point= data.first()
#print "label of first point: %f" % first_point.label
#print "features of first point: %s" % str(first_point.features)
#print "feature vector length: %d" % len(first_point.features)

lrModel=LinearRegressionWithSGD.train(data, iterations=10, step=0.1, intercept=False)
actual_vs_pred=data.map(lambda p: (p.label, lrModel.predict(p.features)))
print actual_vs_pred.take(10)
