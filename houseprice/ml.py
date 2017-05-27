from pyspark import SparkContext
import numpy as np

train_file_path="/usr/bigdata/data/houseprice/noheader_train.csv"

sc=SparkContext("local[2]","spark house price app")
