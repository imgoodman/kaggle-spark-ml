from pyspark import SparkContext

sc=SparkContext("local[2]","titanic spark app")

input_file="/usr/bigdata/spark/kaggle-spark-ml-app/kaggle-spark-ml/titanic/data/train.csv"

raw_data=sc.textFile(input_file)

#raw_records=raw_data.map(lambda line: line.split(",")).map(lambda (passengerId, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked):(pclass, sex, age, sibsp, parch, ticket, fare, cabin, embarked, survived))
#raw_records=raw_data.map(lambda line: line.split(",")).map(lambda records:(records[2],records[4], records[5],records[6], records[7], records[8], records[9], records[10], records[11], records[1]))
raw_records=raw_data.map(lambda line: line.split(","))
print raw_records.first()
