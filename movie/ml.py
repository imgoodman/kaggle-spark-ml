#-*- coding:utf8-*-
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS,Rating
import matplotlib.pyplot as plt
import numpy as np

train_file_path="/usr/bigdata/data/ml-100k/u.data"
movie_file_path="/usr/bigdata/data/ml-100k/u.item"

sc=SparkContext("local[2]","spark movie app")

raw_data=sc.textFile(train_file_path).map(lambda line: line.split("\t"))


raw_movies=sc.textFile(movie_file_path).map(lambda line: line.split("|"))
#print raw_movies.take(10)
movies=raw_movies.map(lambda fields:(int(fields[0]), fields[1]))
#print movies.take(10)
def getMovieById(movieId):
    return movies.filter(lambda (id,title):id==movieId).collectAsMap()
#print(raw_data.first())
#print(" movie is:" +  str(getMovieById(123)))

"""
评级矩阵是943*1682
其中943是用户数量
1682是电影数量
"""
ratings=raw_data.map(lambda fields: Rating(int(fields[0]), int(fields[1]), float(fields[2]) ))

#print(ratings.take(3))
"""
Train a matrix factorization model given an RDD of ratings by users for a subset of products. The ratings matrix is approximated as the product of two lower-rank matrices of a given rank (number of features).
To solve for these features, ALS is run iteratively with a configurable level of parallelism


ratings---RDD of Rating or (userID,productID,rating) turple Rating对象的RDD，或者由(userID,productID,rating)组成的元祖
rank---Rank of the feature matrices computed (number of features)因子个数，低阶近似矩阵中隐含特征个数。通常，合理取值为10-200
iterations---Number of iterations of ALS (default 5)
lambda---Regularization parameter, default 0.1
blocks---Number of blocks used to parallelize the computation. A value of -1 will use an auto-configured number of blocks. default -1
nonnegative---A value of True will solve least-squares with nonnegativity constraints, default False
seed---Random seed for initial matrix factorization model. A value of None will use system time as the seed, default None

返回MatrixFactorizationModel对象
该对象将用户因子保存在一个（id，factor）对类型的RDD中，称为userFeatures
将物品因子保存在一个（id，factor）对类型的RDD中，称为productFeatures

"""
rank=50
iterations=10
regParam=0.1
model=ALS.train(ratings, rank, iterations, regParam)
"""
将943*1682矩阵进行分解
并且因子个数设置为rank=50了，
那么，分解得到的用户因子矩阵为943*50
产品因子矩阵（也是电影因子矩阵）为1682*50（其实中间有转置的过程）
"""
#print(model.useurFeatures)
#print(model.productFeatures)

#predRating=model.predict(789, 123)
#print(predRating)


"""
为某个用户推荐K个物品
recommendProducts(user, num)
其中,user：用户ID
num：要推荐的物品个数
给指定用户推荐的电影
"""
def recommend():
    userId=789
    K=10
    opKRecs=model.recommendProducts(userId, K)
    print "model recommend movies:"
    print(topKRecs)
    """
    指定用户实际看过的电影
    """
    moviesForUser=ratings.filter(lambda r: r[0]==userId)
    #print moviesForUser.collect()
    topActualMovies=moviesForUser.map(lambda r: (r.product, r.rating)).map(lambda (k,v): (v,k)).sortByKey(ascending=False)
    print("actual movies seen:")
    print topActualMovies.take(10)


"""
物品推荐：给定物品，哪些物品与它相似
常见的相似度衡量方法：
1.皮尔森相关系数Pearson correlation
2.针对实数的余弦相似度cosine similarity，指的是两个向量在n维空间里两者夹角的度数，它是两个向量的点积与各向量范数的乘积的商。取值-1到1，1表示完全相似，0表示两者互不相关，无相似性。-1表示两者不仅不相关，而且完全不相同
3.针对二元向量的杰卡德相似系数Jaccard similarity
"""
def cosine_similarity(vec1, vec2):
    return vec1.dot(vec2)/(vec1.norm() * vec2.norm())
def product_similar():
    from jblas import DoubleMatrix
    itemId=567
    itemFactor=model.productFeatures.filter(lambda (id,factor):id==itemId).map(lambda (id,factor): DoubleMatrix(factor))
    print cosine_similarity(itemFactor, itemFactor)
product_similar()
