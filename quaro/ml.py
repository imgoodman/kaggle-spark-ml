#-*-coding:utf8-*-
from pyspark import SparkContext
import numpy as np
import pandas as pd
import csv
import unicodecsv
import StringIO
#import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

#train_file_path="/usr/bigdata/data/quaro/train_noheader.csv"
train_file_path="/usr/bigdata/data/quaro/train.csv"

"""
通过pandas来探索数据
"""
#读取训练数据
train_df=pd.read_csv(train_file_path)
#print train_df.shape
#print train_df.head().values

sc=SparkContext("local[2]","spark quaro question pairs duplicate app")
raw_data=sc.parallelize(train_df.values)
#print raw_data.take(10)
"""
终于躲开了question中带逗号
从而导致split解析出错的坑了
不过 这个方式还是不好
如果训练数据量较大
通过pandas读取 再传递给spark
就没有发挥spark分布式数据源的优势了
"""
#raw_data=pd.read_csv(train_file_path,encoding="utf-8")

def loadRecord(line):
    """
    解析一行csv记录
    """
    input_line=StringIO.StringIO(line)
    #row=unicodecsv.reader(input_line, encoding="utf-8")
    #return row.next()
    #reader=csv.DictReader(input_line,fieldnames=["id","qid1","qid2","question1","question2","is_duplicate"])
    reader=csv.reader(input_line)
    return reader.next()
    #data=[]
    #for row in reader:
    #    print row
    #    data.append([unicode(cell,"utf-8") for cell in row])
    #return data[0]
    #return reader.next()

#raw_data=sc.textFile(train_file_path).map(loadRecord)
#print raw_data.take(10)
"""
看看总共有多少个问题呢
"""
qid1s=raw_data.map(lambda fields:fields[1])
qid2s=raw_data.map(lambda fields:fields[2])
qids=qid1s.union(qid2s).distinct().count()
print "total questions is: %d" % qids
