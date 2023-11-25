# Nuestra branch unit2 

//PRACTICA 1 U2 EN EQUIPO LOS PERRONES

# Unit2 Practice 1 LINEAR REGRESSION EXERCISE
### Import LinearRegression
```scala
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
```
```sh
scala> import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegression

scala> import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession
```
### Opcional: Utilice el siguiente codigo para configurar errores
```scala
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```
```sh
scala> import org.apache.log4j._
import org.apache.log4j._
scala> Logger.getLogger("org").setLevel(Level.ERROR)
```

### Inicie una simple Sesion Spark
```scala
val spark = SparkSession.builder().getOrCreate()
```
```sh
scala> val spark = SparkSession.builder().getOrCreate()
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@60a0f09f
```

### Utilice Spark para el archivo csv Clean-Ecommerce 
```scala
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")
```
```sh
scala> val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")
data: org.apache.spark.sql.DataFrame = [Email: string, Avatar: string ... 5 more fields]
```
### Imprima el schema en el DataFrame
```scala
data.printSchema()
```
```sh
scala> data.printSchema()
root
 |-- Email: string (nullable = true)
 |-- Avatar: string (nullable = true)
 |-- Avg Session Length: double (nullable = true)
 |-- Time on App: double (nullable = true)
 |-- Time on Website: double (nullable = true)
 |-- Length of Membership: double (nullable = true)
 |-- Yearly Amount Spent: double (nullable = true)
```
### Imprima un renglon de ejemplo del DataFrane.
```scala
data.head(1)
```
```sh
scala> data.head(1)
res2: Array[org.apache.spark.sql.Row] = Array([mstephenson@fernandez.com,Violet,34.49726772511229,12.65565114916675,39.57766801952616,4.0826206329529615,587.9510539684005])
```
### Transforme el data frame para que tome la forma de ("label","features")
### Importe VectorAssembler y Vectors
```scala
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```
```sh
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```

### Renombre la columna Yearly Amount Spent como "label", Tambien de los datos tome solo la columa numerica, Deje todo esto como un nuevo DataFrame que se llame df
```scala
data.columns
val df = data.select(data("Yearly Amount Spent").as("label"), $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership")
```
```sh
val res3: Array[String] = Array(Email, Avatar, Avg Session Length, Time on App, Time on Website, Length of Membership, Yearly Amount Spent)
val df: org.apache.spark.sql.DataFrame = [label: double, Avg Session Length: double ... 3 more fields]
```

### Utilice el objeto VectorAssembler para convertir la columnas de entradas del df, a una sola columna de salida de un arreglo llamado  "features", Configure las columnas de entrada de donde se supone que leemos los valores, Llamar a esto nuevo assambler.
```scala
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership")).setOutputCol("features")
```
```sh
val assembler: org.apache.spark.ml.feature.VectorAssembler = VectorAssembler: uid=vecAssembler_115df6a50100, handleInvalid=error, numInputCols=4
```

### Utilice el assembler para transform nuestro DataFrame a dos columnas: label and features
```scala
val output = assembler.transform(df).select($"label", $"features")
```
```sh
val output: org.apache.spark.sql.DataFrame = [label: double, features: vector]
```

### Crear un objeto para modelo de regresion linea.
```scala
var lr = new LinearRegression()
```
```sh
scala> var lr = new LinearRegression()
lr: org.apache.spark.ml.regression.LinearRegression = linReg_06172a1924ab
```

###  Ajuste el modelo para los datos y llame a este modelo lrModelo
```scala
var lrModelo = lr.fit(output)
```
```sh
scala> var lrModelo = lr.fit(output)
23/11/20 15:17:06 WARN Instrumentation: [959d57c4] regParam is zero, which might cause numerical instability and overfitting.
lrModelo: org.apache.spark.ml.regression.LinearRegressionModel = LinearRegressionModel: uid=linReg_06172a1924ab, numFeatures=4
```

### Imprima the  coefficients y intercept para la regresion lineal
```scala
lrModelo.coefficients
lrModelo.intercept
```
```sh
scala> lrModelo.coefficients
res19: org.apache.spark.ml.linalg.Vector = [25.734271084670716,38.709153810828816,0.43673883558514964,61.57732375487594]

scala> lrModelo.intercept
res20: Double = -1051.5942552990748
```

### Resuma el modelo sobre el conjunto de entrenamiento imprima la salida de algunas metricas!,Utilize metodo .summary de nuestro  modelo para crear un objeto llamado trainingSummary Muestre los valores de residuals, el RMSE, el MSE, y tambien el R^2 .
```scala
val trainingSummary = lrModelo.summary

trainingSummary.residuals.show()
trainingSummary.rootMeanSquaredError
trainingSummary.meanSquaredError
trainingSummary.r2
```
```sh
scala> val trainingSummary = lrModelo.summary
trainingSummary: org.apache.spark.ml.regression.LinearRegressionTrainingSummary = org.apache.spark.ml.regression.LinearRegressionTrainingSummary@626519ef

scala>

scala> trainingSummary.residuals.show()
+-------------------+
|          residuals|
+-------------------+
| -6.788234090018818|
| 11.841128565326073|
| -17.65262700858966|
| 11.454889631178617|
| 7.7833824373080915|
|-1.8347332184773677|
|  4.620232401352382|
| -8.526545950978175|
| 11.012210896516763|
|-13.828032682158891|
| -16.04456458615175|
|  8.786634365463442|
| 10.425717191807507|
| 12.161293785003522|
|  9.989313714461446|
| 10.626662732649379|
|  20.15641408428496|
|-3.7708446586326545|
| -4.129505481591934|
|  9.206694655890487|
+-------------------+
only showing top 20 rows


scala> trainingSummary.rootMeanSquaredError
res30: Double = 9.923256785022229

scala> trainingSummary.meanSquaredError
res31: Double = 98.47102522148971

scala> trainingSummary.r2
res32: Double = 0.9843155370226727
```

# PRACTICA 2 LOGISTIC REGRESION

### Inicia una sesion en Spark
```scala
val spark = SparkSession.builder().getOrCreate()

```
```sh
scala> val spark = SparkSession.builder().getOrCreate()
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@38207d19
```

### Utilice Spark para leer el archivo csv Advertising.
```scala
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")

```
```sh
scala> val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
data: org.apache.spark.sql.DataFrame = [Daily Time Spent on Site: double, Age: int ... 8 more fields]
```

### Imprima el Schema del DataFrame
```scala

data.printSchema()
```
```sh
scala> data.printSchema()
root
 |-- Daily Time Spent on Site: double (nullable = true)
 |-- Age: integer (nullable = true)
 |-- Area Income: double (nullable = true)
 |-- Daily Internet Usage: double (nullable = true)
 |-- Ad Topic Line: string (nullable = true)
 |-- City: string (nullable = true)
 |-- Male: integer (nullable = true)
 |-- Country: string (nullable = true)
 |-- Timestamp: timestamp (nullable = true)
 |-- Clicked on Ad: integer (nullable = true)
```

### Imprimir un renglon de ejemplo

```scala
data.head(1)
```
```sh
scala> data.head(1)
res3: Array[org.apache.spark.sql.Row] = Array([68.95,35,61833.9,256.09,Cloned 5thgeneration orchestration,Wrightburgh,0,Tunisia,2016-03-27 00:53:11.0,0])
```

### Tome la siguientes columnas como features "Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Timestamp","Male"

```scala
val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")
```
```sh
scala> val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")
```

### Cree una nueva clolumna llamada "Hour" del Timestamp conteniendo la  "Hour of the click" 

```scala
val timedata = data.withColumn("Hour",hour(data("Timestamp")))
```
```sh
scala> val timedata = data.withColumn("Hour",hour(data("Timestamp")))
timedata: org.apache.spark.sql.DataFrame = [Daily Time Spent on Site: double, Age: int ... 9 more fields]
```

### Importe VectorAssembler y Vectors
```scala
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```
```sh
scala> import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorAssembler

scala> import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vectors
```

###   Cree un nuevo objecto VectorAssembler llamado assembler para los feature

```scala
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"))
                  .setOutputCol("features"))
```
```sh

scala> val assembler = (new VectorAssembler()
     |                   .setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"))
     |                   .setOutputCol("features"))
assembler: org.apache.spark.ml.feature.VectorAssembler = VectorAssembler: uid=vecAssembler_77900d31765a, handleInvalid=error, numInputCols=6
```

### Utilice randomSplit para crear datos de train y test divididos en 70/30   

```scala
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)
```
```sh
scala> val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)
training: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, Daily Time Spent on Site: double ... 5 more fields]
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, Daily Time Spent on Site: double ... 5 more fields]
```

### Importe  Pipeline  

```scala
import org.apache.spark.ml.Pipeline
```
```sh
scala> import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.Pipeline
```

###  Cree un nuevo objeto de  LogisticRegression llamado lr 

```scala
val lr = new LogisticRegression()
```
```sh
scala> val lr = new LogisticRegression()
lr: org.apache.spark.ml.classification.LogisticRegression = logreg_93302a572127
```

###  Cree un nuevo  pipeline con los elementos: assembler, lr

```scala
val pipeline = new Pipeline().setStages(Array(assembler, lr))
```
```sh
scala> val pipeline = new Pipeline().setStages(Array(assembler, lr))
pipeline: org.apache.spark.ml.Pipeline = pipeline_0b4b2e9e5aae
```

###  Ajuste (fit) el pipeline para el conjunto de training.

```scala
val model = pipeline.fit(training)
```
```sh
scala> val model = pipeline.fit(training)
23/11/24 19:09:34 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS
model: org.apache.spark.ml.PipelineModel = pipeline_0b4b2e9e5aae
```

###  Tome los Resultados en el conjuto Test con transform

```scala
val results = model.transform(test)
```
```sh
scala> val results = model.transform(test)
results: org.apache.spark.sql.DataFrame = [label: int, Daily Time Spent on Site: double ... 9 more fields]
```
###  Para Metrics y Evaluation importe MulticlassMetrics

```scala
import org.apache.spark.mllib.evaluation.MulticlassMetrics
```
```sh
scala> import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
```

###  Convierta los resutalos de prueba (test) en RDD utilizando .as y .rdd

```scala
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
```
```sh
scala> val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
predictionAndLabels: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[65] at rdd at <console>:32
```

###  Inicialice un objeto MulticlassMetrics 

```scala
val metrics = new MulticlassMetrics(predictionAndLabels)
```
```sh
scala> val metrics = new MulticlassMetrics(predictionAndLabels)
metrics: org.apache.spark.mllib.evaluation.MulticlassMetrics = org.apache.spark.mllib.evaluation.MulticlassMetrics@7904ee3c
```

###  Imprima la  Confusion matrix 

```scala
println("Confusion matrix:")
println(metrics.confusionMatrix)
```
```sh
scala> println("Confusion matrix:")
Confusion matrix:

scala> println(metrics.confusionMatrix)
136.0  1.0
4.0    146.0
```

### Accurancy  

```scala
metrics.accuracy
```
```sh
scala> metrics.accuracy
res6: Double = 0.9825783972125436
```
# PRACTICA 3 DECISON TREE CLASISFFIER CON SCALA Y SPARK
 ```sh
 scala> :load DECISIONTREECLASSIFICATIONMODEL.scala
Loading DECISIONTREECLASSIFICATIONMODEL.scala...
DECISIONTREECLASSIFICATIONMODEL.scala:1: error: illegal start of definition
package org.apache.spark.examples.ml
^
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession
defined object DecisionTreeClassificationExample

scala> // scalastyle:off println

scala> package org.apache.spark.examples.ml
<console>:1: error: illegal start of definition
       package org.apache.spark.examples.ml
       ^

scala> 

scala> // $example on$

scala> import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.Pipeline

scala> import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassificationModel

scala> import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassifier

scala> import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

scala> import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

scala> // $example off$

scala> import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession

scala> val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()
23/11/22 19:17:28 WARN SparkSession: Using an existing Spark session; only runtime SQL configurations will take effect.
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@55dfdcd0

scala> val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
23/11/22 19:17:42 WARN LibSVMFileFormat: 'numFeatures' option not specified, determining the number of features by going though the input. If you know the number in 
advance, please specify it via 'numFeatures' option to avoid the extra scan.
data: org.apache.spark.sql.DataFrame = [label: double, features: vector]

scala> data.show()
+-----+--------------------+
|label|            features|
+-----+--------------------+
|  0.0|(692,[127,128,129...|
|  1.0|(692,[158,159,160...|
|  1.0|(692,[124,125,126...|
|  1.0|(692,[152,153,154...|
|  1.0|(692,[151,152,153...|
|  0.0|(692,[129,130,131...|
|  1.0|(692,[158,159,160...|
|  1.0|(692,[99,100,101,...|
|  0.0|(692,[154,155,156...|
|  0.0|(692,[127,128,129...|
|  1.0|(692,[154,155,156...|
|  0.0|(692,[153,154,155...|
|  0.0|(692,[151,152,153...|
|  1.0|(692,[129,130,131...|
|  0.0|(692,[154,155,156...|
|  1.0|(692,[150,151,152...|
|  0.0|(692,[124,125,126...|
|  0.0|(692,[152,153,154...|
|  1.0|(692,[97,98,99,12...|
|  1.0|(692,[124,125,126...|
+-----+--------------------+
only showing top 20 rows


scala> val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = StringIndexerModel: uid=strIdx_8b88fc05613f, handleInvalid=error

scala> val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data) 
featureIndexer: org.apache.spark.ml.feature.VectorIndexerModel = VectorIndexerModel: uid=vecIdx_3d9dc394fa28, numFeatures=692, handleInvalid=error

scala> val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]

scala> val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
dt: org.apache.spark.ml.classification.DecisionTreeClassifier = dtc_ddea68a6b83e

scala> val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labelsArray(0))
labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_a0e2185e1730

scala> val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
pipeline: org.apache.spark.ml.Pipeline = pipeline_89903fa347b0

scala>  val model = pipeline.fit(trainingData)
model: org.apache.spark.ml.PipelineModel = pipeline_89903fa347b0

scala>

scala> val predictions = model.transform(testData)
predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 6 more fields]

scala> prediction.show()
<console>:36: error: not found: value prediction
       prediction.show()
       ^

scala> predictions.show()
+-----+--------------------+------------+--------------------+-------------+-----------+----------+--------------+
|label|            features|indexedLabel|     indexedFeatures|rawPrediction|probability|prediction|predictedLabel|
+-----+--------------------+------------+--------------------+-------------+-----------+----------+--------------+
|  0.0|(692,[124,125,126...|         1.0|(692,[124,125,126...|   [0.0,34.0]|  [0.0,1.0]|       1.0|           0.0|
|  0.0|(692,[126,127,128...|         1.0|(692,[126,127,128...|   [0.0,34.0]|  [0.0,1.0]|       1.0|           0.0|
|  0.0|(692,[127,128,129...|         1.0|(692,[127,128,129...|   [0.0,34.0]|  [0.0,1.0]|       1.0|           0.0|
|  0.0|(692,[128,129,130...|         1.0|(692,[128,129,130...|   [0.0,34.0]|  [0.0,1.0]|       1.0|           0.0|
|  0.0|(692,[151,152,153...|         1.0|(692,[151,152,153...|   [0.0,34.0]|  [0.0,1.0]|       1.0|           0.0|
|  0.0|(692,[153,154,155...|         1.0|(692,[153,154,155...|   [0.0,34.0]|  [0.0,1.0]|       1.0|           0.0|
|  0.0|(692,[154,155,156...|         1.0|(692,[154,155,156...|   [39.0,0.0]|  [1.0,0.0]|       0.0|           1.0|
|  0.0|(692,[155,156,180...|         1.0|(692,[155,156,180...|   [0.0,34.0]|  [0.0,1.0]|       1.0|           0.0|
|  0.0|(692,[234,235,237...|         1.0|(692,[234,235,237...|   [0.0,34.0]|  [0.0,1.0]|       1.0|           0.0|
|  1.0|(692,[99,100,101,...|         0.0|(692,[99,100,101,...|   [0.0,34.0]|  [0.0,1.0]|       1.0|           0.0|
|  1.0|(692,[123,124,125...|         0.0|(692,[123,124,125...|   [39.0,0.0]|  [1.0,0.0]|       0.0|           1.0|
|  1.0|(692,[124,125,126...|         0.0|(692,[124,125,126...|   [39.0,0.0]|  [1.0,0.0]|       0.0|           1.0|
|  1.0|(692,[126,127,128...|         0.0|(692,[126,127,128...|   [39.0,0.0]|  [1.0,0.0]|       0.0|           1.0|
|  1.0|(692,[126,127,128...|         0.0|(692,[126,127,128...|   [39.0,0.0]|  [1.0,0.0]|       0.0|           1.0|
|  1.0|(692,[126,127,128...|         0.0|(692,[126,127,128...|   [39.0,0.0]|  [1.0,0.0]|       0.0|           1.0|
|  1.0|(692,[127,128,155...|         0.0|(692,[127,128,155...|   [39.0,0.0]|  [1.0,0.0]|       0.0|           1.0|
|  1.0|(692,[128,129,130...|         0.0|(692,[128,129,130...|   [39.0,0.0]|  [1.0,0.0]|       0.0|           1.0|
|  1.0|(692,[128,129,130...|         0.0|(692,[128,129,130...|   [39.0,0.0]|  [1.0,0.0]|       0.0|           1.0|
|  1.0|(692,[128,129,130...|         0.0|(692,[128,129,130...|   [39.0,0.0]|  [1.0,0.0]|       0.0|           1.0|
|  1.0|(692,[129,130,131...|         0.0|(692,[129,130,131...|   [39.0,0.0]|  [1.0,0.0]|       0.0|           1.0|
+-----+--------------------+------------+--------------------+-------------+-----------+----------+--------------+
only showing top 20 rows


scala> predictions.select("predictedLabel", "label", "features").show(5)
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[126,127,128...|
|           0.0|  0.0|(692,[127,128,129...|
|           0.0|  0.0|(692,[128,129,130...|
|           0.0|  0.0|(692,[151,152,153...|
+--------------+-----+--------------------+
only showing top 5 rows


scala> val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = MulticlassClassificationEvaluator: uid=mcEval_9d4488f78cc3, metricName=accuracy, metricLabel=0.0, beta=1.0, eps=1.0E-15

scala>     val accuracy = evaluator.evaluate(predictions)
accuracy: Double = 0.9259259259259259

scala>     println(s"Test Error = ${(1.0 - accuracy)}")
Test Error = 0.07407407407407407

scala> val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
treeModel: org.apache.spark.ml.classification.DecisionTreeClassificationModel = DecisionTreeClassificationModel: uid=dtc_ddea68a6b83e, depth=1, numNodes=3, numClasses=2, numFeatures=692

scala>     println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
Learned classification tree model:
 DecisionTreeClassificationModel: uid=dtc_ddea68a6b83e, depth=1, numNodes=3, numClasses=2, numFeatures=692
  If (feature 406 <= 22.0)
   Predict: 1.0
  Else (feature 406 > 22.0)
   Predict: 0.0


scala> val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
treeModel: org.apache.spark.ml.classification.DecisionTreeClassificationModel = DecisionTreeClassificationModel: uid=dtc_ddea68a6b83e, depth=1, numNodes=3, numClasses=2, numFeatures=692

scala>     println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

 ```
 # PRACTICA 4 PERCEPTRON MULTYLAYER CLASSIFIER 

 ```sh
 Using Scala version 2.12.17 (Java HotSpot(TM) 64-Bit Server VM, Java 11.0.19)
Type in expressions to have them evaluated.
Type :help for more information.

scala> import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier

scala> import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

scala>

scala> // Load the data stored in LIBSVM format as a DataFrame.

scala> val data = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")
23/11/22 19:49:23 WARN LibSVMFileFormat: 'numFeatures' option not specified, determining the number of features by going though the input. If you know the number in advance, please specify it via 'numFeatures' option to avoid the extra scan.
data: org.apache.spark.sql.DataFrame = [label: double, features: vector]

scala> data.show()
+-----+--------------------+
|label|            features|
+-----+--------------------+
|  1.0|(4,[0,1,2,3],[-0....|
|  1.0|(4,[0,1,2,3],[-0....|
|  1.0|(4,[0,1,2,3],[-0....|
|  1.0|(4,[0,1,2,3],[-0....|
|  0.0|(4,[0,1,2,3],[0.1...|
|  1.0|(4,[0,2,3],[-0.83...|
|  2.0|(4,[0,1,2,3],[-1....|
|  2.0|(4,[0,1,2,3],[-1....|
|  1.0|(4,[0,1,2,3],[-0....|
|  0.0|(4,[0,2,3],[0.611...|
|  0.0|(4,[0,1,2,3],[0.2...|
|  1.0|(4,[0,1,2,3],[-0....|
|  1.0|(4,[0,1,2,3],[-0....|
|  2.0|(4,[0,1,2,3],[-0....|
|  2.0|(4,[0,1,2,3],[-0....|
|  2.0|(4,[0,1,2,3],[-0....|
|  1.0|(4,[0,2,3],[-0.94...|
|  2.0|(4,[0,1,2,3],[-0....|
|  0.0|(4,[0,1,2,3],[0.1...|
|  2.0|(4,[0,1,2,3],[-0....|
+-----+--------------------+
only showing top 20 rows


scala> val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
splits: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([label: double, features: vector], [label: double, features: vector])

scala> val train = splits(0)
train: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]

scala> val test = splits(1)
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]

scala> val layers = Array[Int](4, 5, 4, 3)
layers: Array[Int] = Array(4, 5, 4, 3)

scala> val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_569c7e8f9eb9

scala> val model = trainer.fit(train)
model: org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel = MultilayerPerceptronClassificationModel: uid=mlpc_569c7e8f9eb9, numLayers=4, numClasses=3, numFeatures=4

scala> val result = model.transform(test)
result: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 3 more fields]

scala> val predictionAndLabels = result.select("prediction", "label")
predictionAndLabels: org.apache.spark.sql.DataFrame = [prediction: double, label: double]

scala> predictionAndLabels.show()
+----------+-----+
|prediction|label|
+----------+-----+
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       2.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       0.0|  0.0|
+----------+-----+
only showing top 20 rows


scala> val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = MulticlassClassificationEvaluator: uid=mcEval_87c2fa45d321, metricName=accuracy, metricLabel=0.0, beta=1.0, eps=1.0E-15

scala> println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
Test set accuracy = 0.9523809523809523

 ``` 

 # PRACTICA EVALUATORIA 

### Librerias
 ```SH 
 scala> import org.apache.spark.ml.classification.MultilayerPerceptronClassifier 
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier

scala> import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

scala> import org.apache.spark.ml.feature.IndexToString 
import org.apache.spark.ml.feature.IndexToString

scala> import org.apache.spark.ml.feature.StringIndexer 
import org.apache.spark.ml.feature.StringIndexer

scala> import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorIndexer

scala> import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorAssembler

scala> import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.Pipeline

scala>
```

### Cargar el archivo iris

```sh
scala> var iris = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")
iris: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 3 more fields]
```

### Mostrar el esquema (muestra los primeros 20 renglones)

```scala
iris.show() 
```

```sh
scala> iris.show() 
+------------+-----------+------------+-----------+-------+
|sepal_length|sepal_width|petal_length|petal_width|species|
+------------+-----------+------------+-----------+-------+
|         5.1|        3.5|         1.4|        0.2| setosa|
|         4.9|        3.0|         1.4|        0.2| setosa|
|         4.7|        3.2|         1.3|        0.2| setosa|
|         4.6|        3.1|         1.5|        0.2| setosa|
|         5.0|        3.6|         1.4|        0.2| setosa|
|         5.4|        3.9|         1.7|        0.4| setosa|
|         4.6|        3.4|         1.4|        0.3| setosa|
|         5.0|        3.4|         1.5|        0.2| setosa|
|         4.4|        2.9|         1.4|        0.2| setosa|
|         4.9|        3.1|         1.5|        0.1| setosa|
|         5.4|        3.7|         1.5|        0.2| setosa|
|         4.8|        3.4|         1.6|        0.2| setosa|
|         4.8|        3.0|         1.4|        0.1| setosa|
|         4.3|        3.0|         1.1|        0.1| setosa|
|         5.8|        4.0|         1.2|        0.2| setosa|
|         5.7|        4.4|         1.5|        0.4| setosa|
|         5.4|        3.9|         1.3|        0.4| setosa|
|         5.1|        3.5|         1.4|        0.3| setosa|
|         5.7|        3.8|         1.7|        0.3| setosa|
|         5.1|        3.8|         1.5|        0.3| setosa|
+------------+-----------+------------+-----------+-------+
only showing top 20 rows

```
### Imprimir el esquema

```scala
iris.printSchema()
```

```sh
scala> iris.printSchema()
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)

```

### Imprime los primeros 5 renglones

``` scala
iris.show(5) 
```

```sh
scala> iris.show(5)
+------------+-----------+------------+-----------+-------+
|sepal_length|sepal_width|petal_length|petal_width|species|
+------------+-----------+------------+-----------+-------+
|         5.1|        3.5|         1.4|        0.2| setosa|
|         4.9|        3.0|         1.4|        0.2| setosa|
|         4.7|        3.2|         1.3|        0.2| setosa|
|         4.6|        3.1|         1.5|        0.2| setosa|
|         5.0|        3.6|         1.4|        0.2| setosa|
+------------+-----------+------------+-----------+-------+
only showing top 5 rows

```
### Mostrar la descripcion del esquema
```scala
iris.describe().show()
```

```sh

scala> iris.describe().show()
23/11/22 20:25:05 WARN package: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.
+-------+------------------+-------------------+------------------+------------------+---------+
|summary|      sepal_length|        sepal_width|      petal_length|       petal_width|  species|
+-------+------------------+-------------------+------------------+------------------+---------+
|  count|               150|                150|               150|               150|      150|
|   mean| 5.843333333333335| 3.0540000000000007|3.7586666666666693|1.1986666666666672|     null|
| stddev|0.8280661279778637|0.43359431136217375| 1.764420419952262|0.7631607417008414|     null|
|    min|               4.3|                2.0|               1.0|               0.1|   setosa|
|    max|               7.9|                4.4|               6.9|               2.5|virginica|
+-------+------------------+-------------------+------------------+------------------+---------+
```
### Haga la transformacion pertinente para los datos categoricos los cuales seran nuestras etiquetas a clasificar

```scala
val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width")).setOutputCol("features")
```

```sh
scala> val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width")).setOutputCol("features")   
assembler: org.apache.spark.ml.feature.VectorAssembler = VectorAssembler: uid=vecAssembler_fc2de4a451a0, handleInvalid=error, numInputCols=4

scala> val features = assembler.transform(iris)
features: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 4 more fields]
```
scala> features.show()
+------------+-----------+------------+-----------+-------+-----------------+
|sepal_length|sepal_width|petal_length|petal_width|species|         features|
+------------+-----------+------------+-----------+-------+-----------------+
|         5.1|        3.5|         1.4|        0.2| setosa|[5.1,3.5,1.4,0.2]|
|         4.9|        3.0|         1.4|        0.2| setosa|[4.9,3.0,1.4,0.2]|
|         4.7|        3.2|         1.3|        0.2| setosa|[4.7,3.2,1.3,0.2]|
|         4.6|        3.1|         1.5|        0.2| setosa|[4.6,3.1,1.5,0.2]|
|         5.0|        3.6|         1.4|        0.2| setosa|[5.0,3.6,1.4,0.2]|
|         5.4|        3.9|         1.7|        0.4| setosa|[5.4,3.9,1.7,0.4]|
|         4.6|        3.4|         1.4|        0.3| setosa|[4.6,3.4,1.4,0.3]|
|         5.0|        3.4|         1.5|        0.2| setosa|[5.0,3.4,1.5,0.2]|
|         4.4|        2.9|         1.4|        0.2| setosa|[4.4,2.9,1.4,0.2]|
|         4.9|        3.1|         1.5|        0.1| setosa|[4.9,3.1,1.5,0.1]|
|         5.4|        3.7|         1.5|        0.2| setosa|[5.4,3.7,1.5,0.2]|
|         4.8|        3.4|         1.6|        0.2| setosa|[4.8,3.4,1.6,0.2]|
|         4.8|        3.0|         1.4|        0.1| setosa|[4.8,3.0,1.4,0.1]|
|         4.3|        3.0|         1.1|        0.1| setosa|[4.3,3.0,1.1,0.1]|
|         5.8|        4.0|         1.2|        0.2| setosa|[5.8,4.0,1.2,0.2]|
|         5.7|        4.4|         1.5|        0.4| setosa|[5.7,4.4,1.5,0.4]|
|         5.4|        3.9|         1.3|        0.4| setosa|[5.4,3.9,1.3,0.4]|
|         5.1|        3.5|         1.4|        0.3| setosa|[5.1,3.5,1.4,0.3]|
|         5.7|        3.8|         1.7|        0.3| setosa|[5.7,3.8,1.7,0.3]|
|         5.1|        3.8|         1.5|        0.3| setosa|[5.1,3.8,1.5,0.3]|
+------------+-----------+------------+-----------+-------+-----------------+
only showing top 20 rows


scala> val indexerLabel = new StringIndexer().setInputCol("species").setOutputCol("indexedLabel").fit(features)
indexerLabel: org.apache.spark.ml.feature.StringIndexerModel = StringIndexerModel: uid=strIdx_2bfb76705984, handleInvalid=error

scala> val indexerFeatures = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)
indexerFeatures: org.apache.spark.ml.feature.VectorIndexer = vecIdx_b24f21534069

scala> val Array(training, test) = features.randomSplit(Array(0.7, 0.3), seed = 12345)
training: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [sepal_length: double, sepal_width: double ... 4 more fields]
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [sepal_length: double, sepal_width: double ... 4 more fields]

scala> training.show()
+------------+-----------+------------+-----------+----------+-----------------+
|sepal_length|sepal_width|petal_length|petal_width|   species|         features|
+------------+-----------+------------+-----------+----------+-----------------+
|         4.3|        3.0|         1.1|        0.1|    setosa|[4.3,3.0,1.1,0.1]|
|         4.4|        2.9|         1.4|        0.2|    setosa|[4.4,2.9,1.4,0.2]|
|         4.4|        3.0|         1.3|        0.2|    setosa|[4.4,3.0,1.3,0.2]|
|         4.4|        3.2|         1.3|        0.2|    setosa|[4.4,3.2,1.3,0.2]|
|         4.5|        2.3|         1.3|        0.3|    setosa|[4.5,2.3,1.3,0.3]|
|         4.6|        3.1|         1.5|        0.2|    setosa|[4.6,3.1,1.5,0.2]|
|         4.6|        3.4|         1.4|        0.3|    setosa|[4.6,3.4,1.4,0.3]|
|         4.6|        3.6|         1.0|        0.2|    setosa|[4.6,3.6,1.0,0.2]|
|         4.7|        3.2|         1.3|        0.2|    setosa|[4.7,3.2,1.3,0.2]|
|         4.7|        3.2|         1.6|        0.2|    setosa|[4.7,3.2,1.6,0.2]|
|         4.8|        3.0|         1.4|        0.1|    setosa|[4.8,3.0,1.4,0.1]|
|         4.8|        3.0|         1.4|        0.3|    setosa|[4.8,3.0,1.4,0.3]|
|         4.8|        3.4|         1.6|        0.2|    setosa|[4.8,3.4,1.6,0.2]|
|         4.8|        3.4|         1.9|        0.2|    setosa|[4.8,3.4,1.9,0.2]|
|         4.9|        2.4|         3.3|        1.0|versicolor|[4.9,2.4,3.3,1.0]|
|         4.9|        3.0|         1.4|        0.2|    setosa|[4.9,3.0,1.4,0.2]|
|         4.9|        3.1|         1.5|        0.1|    setosa|[4.9,3.1,1.5,0.1]|
|         4.9|        3.1|         1.5|        0.1|    setosa|[4.9,3.1,1.5,0.1]|
|         4.9|        3.1|         1.5|        0.1|    setosa|[4.9,3.1,1.5,0.1]|
|         5.0|        2.0|         3.5|        1.0|versicolor|[5.0,2.0,3.5,1.0]|
+------------+-----------+------------+-----------+----------+-----------------+
only showing top 20 rows


scala> test.show()
+------------+-----------+------------+-----------+----------+-----------------+
|sepal_length|sepal_width|petal_length|petal_width|   species|         features|
+------------+-----------+------------+-----------+----------+-----------------+
|         4.6|        3.2|         1.4|        0.2|    setosa|[4.6,3.2,1.4,0.2]|
|         4.8|        3.1|         1.6|        0.2|    setosa|[4.8,3.1,1.6,0.2]|
|         4.9|        2.5|         4.5|        1.7| virginica|[4.9,2.5,4.5,1.7]|
|         5.0|        3.0|         1.6|        0.2|    setosa|[5.0,3.0,1.6,0.2]|
|         5.0|        3.2|         1.2|        0.2|    setosa|[5.0,3.2,1.2,0.2]|
|         5.0|        3.5|         1.3|        0.3|    setosa|[5.0,3.5,1.3,0.3]|
|         5.1|        3.5|         1.4|        0.3|    setosa|[5.1,3.5,1.4,0.3]|
|         5.4|        3.4|         1.5|        0.4|    setosa|[5.4,3.4,1.5,0.4]|
|         5.4|        3.9|         1.3|        0.4|    setosa|[5.4,3.9,1.3,0.4]|
|         5.7|        2.8|         4.1|        1.3|versicolor|[5.7,2.8,4.1,1.3]|
|         5.7|        4.4|         1.5|        0.4|    setosa|[5.7,4.4,1.5,0.4]|
|         5.8|        4.0|         1.2|        0.2|    setosa|[5.8,4.0,1.2,0.2]|
|         6.0|        2.9|         4.5|        1.5|versicolor|[6.0,2.9,4.5,1.5]|
|         6.1|        2.6|         5.6|        1.4| virginica|[6.1,2.6,5.6,1.4]|
|         6.1|        2.9|         4.7|        1.4|versicolor|[6.1,2.9,4.7,1.4]|
|         6.2|        2.2|         4.5|        1.5|versicolor|[6.2,2.2,4.5,1.5]|
|         6.2|        2.9|         4.3|        1.3|versicolor|[6.2,2.9,4.3,1.3]|
|         6.2|        3.4|         5.4|        2.3| virginica|[6.2,3.4,5.4,2.3]|
|         6.4|        2.8|         5.6|        2.2| virginica|[6.4,2.8,5.6,2.2]|
|         6.4|        3.1|         5.5|        1.8| virginica|[6.4,3.1,5.5,1.8]|
+------------+-----------+------------+-----------+----------+-----------------+
only showing top 20 rows


scala> val layers = Array[Int](4, 5, 4, 3)
layers: Array[Int] = Array(4, 5, 4, 3)

scala> val converterLabel = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(indexerLabel.labels)
warning: one deprecation (since 3.0.0); for details, enable `:setting -deprecation' or `:replay -deprecation'
converterLabel: org.apache.spark.ml.feature.IndexToString = idxToStr_0ebe78c0d548

scala>

scala> val pipeline = new Pipeline().setStages(Array(indexerLabel, indexerFeatures, trainer, converterLabel))
<console>:32: error: not found: value trainer
       val pipeline = new Pipeline().setStages(Array(indexerLabel, indexerFeatures, trainer, converterLabel))
                                                                                    ^

scala> 

scala> val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = MulticlassClassificationEvaluator: uid=mcEval_17e839bf29b0, metricName=accuracy, metricLabel=0.0, beta=1.0, eps=1.0E-15

scala> val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(1234).setMaxIter(100)
trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_4ba8d960abf3

scala> val converterLabel = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(indexerLabel.labels)
warning: one deprecation (since 3.0.0); for details, enable `:setting -deprecation' or `:replay -deprecation'
converterLabel: org.apache.spark.ml.feature.IndexToString = idxToStr_ea258fa7ac7c

scala> val pipeline = new Pipeline().setStages(Array(indexerLabel, indexerFeatures, trainer, converterLabel))
pipeline: org.apache.spark.ml.Pipeline = pipeline_eddf478e8197

scala>

scala> val model = pipeline.fit(training)
model: org.apache.spark.ml.PipelineModel = pipeline_eddf478e8197

scala>

scala> val results = model.transform(test)
results: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 10 more fields]

scala>

scala> val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = MulticlassClassificationEvaluator: uid=mcEval_2c06f0c1f79c, metricName=accuracy, metricLabel=0.0, beta=1.0, eps=1.0E-15

scala>

scala> results.show()
+------------+-----------+------------+-----------+----------+-----------------+------------+-----------------+--------------------+--------------------+----------+--------------+
|sepal_length|sepal_width|petal_length|petal_width|   species|         features|indexedLabel|  indexedFeatures|       rawPrediction|         probability|prediction|predictedLabel|
+------------+-----------+------------+-----------+----------+-----------------+------------+-----------------+--------------------+--------------------+----------+--------------+
|         4.6|        3.2|         1.4|        0.2|    setosa|[4.6,3.2,1.4,0.2]|         0.0|[4.6,3.2,1.4,0.2]|[116.639665126747...|[1.0,5.4813128786...| 
      0.0|        setosa|
|         4.8|        3.1|         1.6|        0.2|    setosa|[4.8,3.1,1.6,0.2]|         0.0|[4.8,3.1,1.6,0.2]|[116.639637156628...|[1.0,5.4814587987...| 
      0.0|        setosa|
|         4.9|        2.5|         4.5|        1.7| virginica|[4.9,2.5,4.5,1.7]|         2.0|[4.9,2.5,4.5,1.7]|[-84.649575188505...|[6.45024203756692...| 
      2.0|     virginica|
|         5.0|        3.0|         1.6|        0.2|    setosa|[5.0,3.0,1.6,0.2]|         0.0|[5.0,3.0,1.6,0.2]|[116.639633357211...|[1.0,5.4814786205...| 
      0.0|        setosa|
|         5.0|        3.2|         1.2|        0.2|    setosa|[5.0,3.2,1.2,0.2]|         0.0|[5.0,3.2,1.2,0.2]|[116.639692676539...|[1.0,5.4811691552...| 
      0.0|        setosa|
|         5.0|        3.5|         1.3|        0.3|    setosa|[5.0,3.5,1.3,0.3]|         0.0|[5.0,3.5,1.3,0.3]|[116.639690022570...|[1.0,5.4811830004...| 
      0.0|        setosa|
|         5.1|        3.5|         1.4|        0.3|    setosa|[5.1,3.5,1.4,0.3]|         0.0|[5.1,3.5,1.4,0.3]|[116.639670084502...|[1.0,5.4812870144...| 
      0.0|        setosa|
|         5.4|        3.4|         1.5|        0.4|    setosa|[5.4,3.4,1.5,0.4]|         0.0|[5.4,3.4,1.5,0.4]|[116.639643527038...|[1.0,5.4814255639...| 
      0.0|        setosa|
|         5.4|        3.9|         1.3|        0.4|    setosa|[5.4,3.9,1.3,0.4]|         0.0|[5.4,3.9,1.3,0.4]|[116.639706910982...|[1.0,5.4810948975...| 
      0.0|        setosa|
|         5.7|        2.8|         4.1|        1.3|versicolor|[5.7,2.8,4.1,1.3]|         1.0|[5.7,2.8,4.1,1.3]|[-59.636360064724...|[2.53501472009233...| 
      1.0|    versicolor|
|         5.7|        4.4|         1.5|        0.4|    setosa|[5.7,4.4,1.5,0.4]|         0.0|[5.7,4.4,1.5,0.4]|[116.639701930067...|[1.0,5.4811208816...| 
      0.0|        setosa|
|         5.8|        4.0|         1.2|        0.2|    setosa|[5.8,4.0,1.2,0.2]|         0.0|[5.8,4.0,1.2,0.2]|[116.639739592847...|[1.0,5.4809244080...| 
      0.0|        setosa|
|         6.0|        2.9|         4.5|        1.5|versicolor|[6.0,2.9,4.5,1.5]|         1.0|[6.0,2.9,4.5,1.5]|[-59.636360400803...|[2.53501390922044...| 
      1.0|    versicolor|
|         6.1|        2.6|         5.6|        1.4| virginica|[6.1,2.6,5.6,1.4]|         2.0|[6.1,2.6,5.6,1.4]|[-84.649575188505...|[6.45024203756692...| 
      2.0|     virginica|
|         6.1|        2.9|         4.7|        1.4|versicolor|[6.1,2.9,4.7,1.4]|         1.0|[6.1,2.9,4.7,1.4]|[-59.636375751631...|[2.53497687192180...|       1.0|    versicolor|olor|                                                                                                                                                                         nica|
|         6.2|        2.2|         4.5|        1.5|versicolor|[6.2,2.2,4.5,1.5]|         1.0|[6.2,2.2,4.5,1.5]|[-84.649575188468...|[6.45024203806225...|       2.0|     virgiolor|nica|                                                                                                                                                                         nica|
|         6.2|        2.9|         4.3|        1.3|versicolor|[6.2,2.9,4.3,1.3]|         1.0|[6.2,2.9,4.3,1.3]|[-59.636360064808...|[2.53501471989081...|       1.0|    versicnica|olor|                                                                                                                                                                         nica|
|         6.2|        3.4|         5.4|        2.3| virginica|[6.2,3.4,5.4,2.3]|         2.0|[6.2,3.4,5.4,2.3]|[-84.649575188505...|[6.45024203756692...|       2.0|     virgi----+nica|
|         6.4|        2.8|         5.6|        2.2| virginica|[6.4,2.8,5.6,2.2]|         2.0|[6.4,2.8,5.6,2.2]|[-84.649575188505...|[6.45024203756692...|       2.0|     virginica|
|         6.4|        3.1|         5.5|        1.8| virginica|[6.4,3.1,5.5,1.8]|         2.0|[6.4,3.1,5.5,1.8]|[-84.649575188505...|[6.45024203756692...|       2.0|     virginica|
+------------+-----------+------------+-----------+----------+-----------------+------------+-----------------+--------------------+--------------------+----------+--------------+
only showing top 20 rows


scala> val accuracy = evaluator.evaluate(results)
accuracy: Double = 0.9705882352941176

scala>

scala> println("Error = " + (1.0 - accuracy))
Error = 0.02941176470588236

scala>
 
 ```
