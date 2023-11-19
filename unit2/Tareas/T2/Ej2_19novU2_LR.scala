// Importe una  SparkSession con la libreria Logistic Regression
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

// Optional: Utilizar el codigo de  Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Cree un sesion Spark 
val spark = SparkSession.builder().getOrCreate()

// Utilice Spark para leer el archivo csv Advertising.
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")

// Imprima el Schema del DataFrame
data.printSchema()

//Imprimir un renglon de ejemplo
data.head(1)

val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}

//   Hacer lo siguiente:
//    - Renombre la columna "Clicked on Ad" a "label"
//    - Tome la siguientes columnas como features "Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Timestamp","Male"
//    - Cree una nueva clolumna llamada "Hour" del Timestamp conteniendo la  "Hour of the click"

val timedata = data.withColumn("Hour",hour(data("Timestamp")))

val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")

// Importe VectorAssembler y Vectors
// Cree un nuevo objecto VectorAssembler llamado assembler para los feature
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val assembler = (new VectorAssembler()
                  .setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"))
                  .setOutputCol("features"))

// Utilice randomSplit para crear datos de train y test divididos en 70/30
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

// Importe  Pipeline
// Cree un nuevo objeto de  LogisticRegression llamado lr

// Cree un nuevo  pipeline con los elementos: assembler, lr

// Ajuste (fit) el pipeline para el conjunto de training.


// Tome los Resultados en el conjuto Test con transform

import org.apache.spark.ml.Pipeline

val lr = new LogisticRegression()

val pipeline = new Pipeline().setStages(Array(assembler, lr))

val model = pipeline.fit(training)

val results = model.transform(test)


// Para Metrics y Evaluation importe MulticlassMetrics

// Convierta los resutalos de prueba (test) en RDD utilizando .as y .rdd

// Inicialice un objeto MulticlassMetrics 

// Imprima la  Confusion matrix
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

println("Confusion matrix:")
println(metrics.confusionMatrix)

metrics.accuracy