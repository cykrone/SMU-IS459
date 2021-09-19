import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from graphframes import *
from pyspark import SparkConf
from pyspark.context import SparkContext
import pyspark.sql.functions as f
from collections import Counter
from pyspark.sql.types import *
from pyspark.ml.feature import *

def word_count(words):
    words = words.split(" ")
    counted = Counter(words)
    return counted.most_common(20)


spark = SparkSession.builder.appName('sg.edu.smu.is459.assignment2').getOrCreate()

# Load data
posts_df = spark.read.load('/user/remus/parquet-input/hardwarezone.parquet')

# Clean the dataframe by removing rows with any null value
posts_df = posts_df.na.drop()
# posts_df = posts_df.limit(200) 


#posts_df.createOrReplaceTempView("posts")

# Find distinct users
#distinct_author = spark.sql("SELECT DISTINCT author FROM posts")
author_df = posts_df.select('author').distinct()

# print('Author number :' + str(author_df.count()))

# Assign ID to the users
author_id = author_df.withColumn('id', monotonically_increasing_id())
# author_id.show()

# Construct connection between post and author
left_df = posts_df.select('topic', 'author') \
    .withColumnRenamed("topic","ltopic") \
    .withColumnRenamed("author","src_author")

right_df =  left_df.withColumnRenamed('ltopic', 'rtopic') \
    .withColumnRenamed('src_author', 'dst_author')

#  Self join on topic to build connection between authors
author_to_author = left_df. \
    join(right_df, left_df.ltopic == right_df.rtopic) \
    .select(left_df.src_author, right_df.dst_author) \
    .distinct()
edge_num = author_to_author.count()
# print('Number of edges with duplicate : ' + str(edge_num))

# Convert it into ids
id_to_author = author_to_author \
    .join(author_id, author_to_author.src_author == author_id.author) \
    .select(author_to_author.dst_author, author_id.id) \
    .withColumnRenamed('id','src')

id_to_id = id_to_author \
    .join(author_id, id_to_author.dst_author == author_id.author) \
    .select(id_to_author.src, author_id.id) \
    .withColumnRenamed('id', 'dst')

id_to_id = id_to_id.filter(id_to_id.src >= id_to_id.dst).distinct()

id_to_id.cache()

# print("Number of edges without duplciate :" + str(id_to_id.count()))
sc = SparkContext.getOrCreate(SparkConf())
sc.setCheckpointDir("checkpoint")
# Build graph with RDDs
graph = GraphFrame(author_id, id_to_id)
result = graph.connectedComponents()
result.groupBy("component").count().show()


# For complex graph queries, e.g., connected components, you need to set
# the checkopoint directory on HDFS, so Spark can handle failures.
# Remember to change to a valid directory in your HDFS
#spark.sparkContext.setCheckpointDir('/user/remus/spark-checkpoint')
# sc = SparkContext.getOrCreate(SparkConf())
# sc.setCheckpointDir("checkpoint")

# The rest is your work, guys
# ......

#Stopwords remover + Tokenize
tk = RegexTokenizer(pattern=r'(?:\p{Punct}|\s)+', inputCol='content', outputCol='temp1')
sw = StopWordsRemover(inputCol='temp1', outputCol='temp2')
df1 = tk.transform(posts_df)
df2 = sw.transform(df1)
authors_df = df2.withColumn('content', expr('concat_ws(" ", array_distinct(temp2))')) \
            .drop('temp1', 'temp2')

#Combined DF with components, authors, id, and content
combined_df = authors_df.join(result, on=['author'], how='inner')

#Frequency Count in Component
wordfreq_df = combined_df.rdd.map(lambda r: (r.component, r.content)).reduceByKey(lambda x,y: x + y).toDF(['component','content'])
udf_myFunction = udf(word_count, ArrayType(StructType([
    StructField("char", StringType(), False),
    StructField("count", IntegerType(), False)
]))) 
wordfreq_df = wordfreq_df.withColumn("wordfreq", udf_myFunction("content"))
wordfreq_df = wordfreq_df.drop(wordfreq_df.content)

#Display Word Frequency in Community
wordfreq_df.show()

print(wordfreq_df.rdd.take(10))

#TriangleCount
triangles = graph.triangleCount()
combined_triangles = triangles.join(combined_df, on=['author'], how='inner')

avg_triangles = (combined_triangles.rdd
.map(lambda x: (x[6],(x[1],1)))
.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
.map(lambda x: (x[0], x[1][0]/x[1][1]))
.toDF(['component','avgtricount']))

#Display average Triangle count in Community
avg_triangles.orderBy(asc("component")).show()

