# Analyze customer engagement data from an e-commerce company offering both 
# mobile  app  and  website  platforms.  Using  linear  regression,  determine  which 
# platform the company should prioritize for improvement efforts. 
# --------------------------------------------------------------------------------------------------------------

# !pip install pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName("house_price_model").getOrCreate()
df = spark.read.csv("/content/drive/MyDrive/Data Analytics Lab/cruise_ship_info.csv", inferSchema=True, header=True)

indexer = StringIndexer(inputCol="Cruise_line", outputCol="cruise_cat")
indexed = indexer.fit(df).transform(df)

assembler = VectorAssembler(inputCols=["Age", "Tonnage", "passengers", "length", "cabins", "passenger_density", "cruise_cat"], outputCol="features")
output = assembler.transform(indexed)
final_output = output.select("features", "crew")

train_data, test_data = final_output.randomSplit([0.7, 0.3])
model = LinearRegression(featuresCol="features", labelCol="crew")
ship_train_model = model.fit(train_data)
result_model = ship_train_model.evaluate(train_data)
print("Rsquared Error :", result_model.r2)

unlabel_data = test_data.select("features")
predict_value = ship_train_model.transform(unlabel_data)
predict_value.show()