

//Inicia una sesion en Spark
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR) //Quita muchos warnings

val spark = SparkSession.builder().getOrCreate()

// Utilice Spark para el archivo csv Clean-Ecommerce
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")

// Imprima el schema en el DataFrame.
data.printSchema

// Imprima un renglon de ejemplo del DataFrane.
data.head(1)

// Imprima un renglon de ejemplo del DataFrane.
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}
// Transforme el data frame para que tome la forma de
// ("label","features")
// Importe VectorAssembler y Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Renombre la columna Yearly Amount Spent como "label"
//data.columns
var df = data.select(data("Yearly Amount Spent").as("label"), $"Email", $"Avatar", $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership")

// Que el objeto assembler convierta los valores de entrada a un vector
//Transform to vector to the ml algorith can read the input
var assembler = new VectorAssembler().setInputCols(Array("Email", "Avatar", "Avg Session Lenght", "Time on App", "Time on Website","Lenght of Membership")).setOutputCol("features")
var output = assembler.transform(df).select($"label", $"features")

output.show()

//Crear un objeto para modelo de regresion lineal
var lr = new LinearRegression()
var lrModel = lr.fit(output)
var trainingSummary = lrModel.summary

trainingSummary.residuals.show()
trainingSummary.predictions.show()
trainingSummary.r2 //variaza que hay 
trainingSummary.rootMeanSquaredError
