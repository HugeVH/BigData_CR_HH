import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

df.printSchema()
df.show()
df.show(5)
df.describe().show()


//7

val HVRatio = df.withColumn("HV Ratio", $"Volume" / $"High")
HVRatio.show()

//8 Este es el ejercicio 8 
df.select(max("OPEN")).show()

//9 Este es el ejercicio 9
df.select(max("Volume"),min("Volume")).show()

//10 Este es el ejercicio 10
df.select(max("Volume"),min("Volume")).show()

//11_a a) ¿Cuántos días fue la columna “Close” inferior a $ 600?
df.filter($"Close"<600).count()

// b) ¿Qué porcentaje del tiempo fue la columna “High” mayor que $ 500?
(df.filter($"High">500).count()*1.0/df.count())*100

// c) ¿Cuál es la correlación de Pearson entre columna “High” y la columna “Volumen”?
 df.select(corr($"High", $"Volume")).show()

 //d) ¿Cuál es el máximo de la columna “High” por año?
 val df2 = df.withColumn("Year", year(df("Date")))
        df2.groupBy("Year").max("High").show()  


//   e) ¿Cuál es el promedio de la columna “Close” para cada mes del calendario?
        val df3 = df2.withColumn("Month", month(df("Date")))
        df3.groupBy("Month").avg("Close").show()









