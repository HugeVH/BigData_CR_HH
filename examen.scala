import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

df.printSchema()
df.show()
df.show(1)
df.describe().show()

val HVRatio = df.withColumn("HV Ratio", $"Volume" / $"High")
HVRatio.show()

df.select(max("OPEN")).show()
df.select(max("Volume"),min("Volume")).show()

df.filter($"Close"<600).count()

(df.filter($"High">500).count()*1.0/df.count())*100

df.select(corr($"High", $"Volume")).show()

val df2 = df.withColumn("Year", year(df("Date")))
          df2.groupBy("Year").max("High").show()  

 val df3 = df2.withColumn("Month", month(df("Date")))
        df3.groupBy("Month").avg("Close").show()
