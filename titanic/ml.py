from pyspark import SparkContext

sc=SparkContext("local[2]","titanic spark app")

input_file="input/train.csv"

raw_data=sc.textFile(input_file)
