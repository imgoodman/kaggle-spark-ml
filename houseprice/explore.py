#-*-coding:utf8-*-
from pyspark import SparkContext

train_file_path="/usr/bigdata/data/houseprice/noheader_train.csv"

sc=SparkContext("local[2]","spark house price data exploration")

raw_data=sc.textFile(train_file_path).map(lambda line:line.split(","))

"""
总共有多少房子
"""
total_houses=raw_data.count()
print "total %d houses" % total_houses
"""
msubclass
the build class
不知道这个字段是什么意思
"""
msubclass=raw_data.map(lambda fields:fields[1])
print msubclass.countByValue()
