#-*- coding:utf8-*-
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint,LinearRegressionWithSGD
import numpy as np

train_file_path="/usr/bigdata/data/houseprice/noheader_train.csv"

sc=SparkContext("local[2]","spark kaggle house price regression")

raw_data=sc.textFile(train_file_path).map(lambda line:line.split(","))

type_columns=[2,5,7,8,9,10,11,12,13,14,15,16,21,22,23,24,27,28,29,39,40,41,53,55,65,78,79]
type_columns_with_NA=[6,25,30,31,32,33,35,42,57,58,60,63,64,72,73,74,]

number_columns=[1,4,17,18,19,20,34,36,37,38,43,44,45,46,47,48,49,50,51,52,54,56,61,62,66,67,68,69,70,71,75,76,77,]
number_columns_with_NA=[3,26,59,]
number_columns_with_many_zeros=[26,34,36,37,38,44,45,62,66,67,68,69,70,71,75,]

saleprice_column=80

def getMapOfColumn(idx):
    return raw_data.map(lambda fields:fields[idx]).distinct().zipWithIndex().collectAsMap()

def get_type_maps():
    type_maps={}
    for i in type_columns:
        type_maps[i]=getMapOfColumn(i)
    return type_maps

type_maps=get_type_maps()
#print type_maps

def get_type_cnt(maps):
    return sum([len(maps[i]) for i in maps])
    
type_cnt=get_type_cnt(type_maps)
number_cnt=len(number_columns)
total=type_cnt+number_cnt
#print total

def extract_features(fields):
    features=np.zeros(total)
    step=0
    for i in type_columns:
        features[step+ int(type_maps[i][fields[i]]) ]=1.0
        step=step+len(type_maps[i])
    for i in number_columns:
        features[step]=float(fields[i])
        step=step+1
    return features

data=raw_data.map(lambda fields: LabeledPoint(float(fields[saleprice_column]),extract_features(fields)))
print data.first()
