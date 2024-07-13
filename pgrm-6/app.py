# 6. Utilize Apache Spark to achieve the following tasks for the given dataset,
# a)  Find the movie with the lowest average rating with RDD.
# b) Identify users who have rated the most movies.
# c) Explore the distribution of ratings over time.
# d)Find the highest-rated movies with a minimum number of ratings.
# ---------------------------------------------------------------------------------------------

# !pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, year, month
spark = SparkSession.builder.appName("Movie_Rating_Analysis").getOrCreate()

movie_df = spark.read.csv("/content/drive/MyDrive/Data Analytics Lab/movies.csv", inferSchema=True, header=True)
rating_df = spark.read.csv("/content/drive/MyDrive/Data Analytics Lab/ratings.csv", inferSchema=True, header=True)

movie_rdd = movie_df.rdd
rating_rdd = rating_df.rdd

# a) Find the Movie with the Lowest Average Rating Using RDD.
avg_rating = rating_rdd.map( lambda x: ( x["movieId"], (x["rating"], 1) ) )
avg_rating = avg_rating.reduceByKey( lambda x, y: ( x[0] + y[0] , x[1] + y[1] ) )
avg_rating = avg_rating.mapValues( lambda x: x[0]/x[1] )
avg_rating = avg_rating.sortBy( lambda x: x[1] ).first()
print("Movie with the lowest average rating: ", avg_rating)

# b) Identify users who have rated the most movies.
top_rated = rating_rdd.map( lambda x: ( x["userId"], 1) )
top_rated = top_rated.reduceByKey( lambda x, y: x + y )
top_rated = top_rated.sortBy( lambda x: x[0], ascending=False ).top(10)
print("Top users by number of ratings ", top_rated)

# c) Explore the distribution of ratings over time.
rating_time = rating_df.withColumn( "year", year( from_unixtime( rating_df["timestamp"] ) ) )
rating_time = rating_time.withColumn( "month", month( from_unixtime( rating_df["timestamp"] ) ) )
rating_time = rating_time.groupBy( "year", "month" ).count().orderBy("year", "month")
print(rating_time.show())

# d)Find the highest-rated movies with a minimum number of ratings.
high_rated = rating_rdd.map( lambda x: ( x["movieId"], ( x["rating"], 1) ) )
high_rated = high_rated.reduceByKey( lambda x, y: ( x[0] + y[0] , x[1] + y[1] ) )
high_rated = high_rated.mapValues( lambda x: ( x[0]/x[1], x[1] ) )
high_rated = high_rated.filter( lambda x: x[1][1] >= 100 )
high_rated = high_rated.sortBy( lambda x: x[1][0], ascending=False ).take(10)
print(f"Highest-rated movies with at least 100 ratings: {high_rated}")
