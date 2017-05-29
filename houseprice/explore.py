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

def getMapOfColumn(idx):
    return raw_data.map(lambda fields:fields[idx]).distinct().zipWithIndex().collectAsMap()


"""
msubclass
the build class
不知道这个字段是什么意思
it seems that it is enumerated
"""
msubclass=raw_data.map(lambda fields:fields[1])
#print msubclass.countByValue()
"""
try to change to enumerate values
"""
msubclass_map=getMapOfColumn(1)

"""
MSZoning
it seems that it is enumerated
can into types
"""
mszoning=raw_data.map(lambda fields:fields[2])
#print mszoning.countByValue()
mszoning_map=getMapOfColumn(2)

"""
LotFrontage
linear feet of street connected to property
number with NA values
need to fill NA values
"""



"""
LotArea
Lot size in square feet
number
"""

"""
street
tyoe of road access
enumerate: pave or grvl
can into type
"""
street_map=getMapOfColumn(5)

"""
alley
type of alley access
emnurate: pave, grvl
value with NA
"""

"""
LotShape
enumerate 
"""
lotshape_map=getMapOfColumn(7)


"""
LandContour
"""
landcontour_map=getMapOfColumn(8)

"""
Utilities
"""
utilities_map=getMapOfColumn(9)


"""
LotConfig
"""
lotconfig_map=getMapOfColumn(10)

"""
LandSlope
"""
landslope_map=getMapOfColumn(11)


"""
Neighborhood
"""
neighborhood_map=getMapOfColumn(12)

"""
Condition1
"""
condition1_map=getMapOfColumn(13)

"""
Condition2
"""
condition2_map=getMapOfColumn(14)


"""
BldgType
"""
bldgtype_map=getMapOfColumn(15)


"""
HouseType
"""
housetype_map=getMapOfColumn(16)

"""
OverallQual
"""
overallqual_map=getMapOfColumn(17)


