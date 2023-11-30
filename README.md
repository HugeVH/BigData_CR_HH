# PROYECTO FINAL

##### 1.- Librerias 
```scala
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier 
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator 
import org.apache.spark.ml.feature.IndexToString 
import org.apache.spark.ml.feature.StringIndexer 
import org.apache.spark.ml.feature.VectorIndexer 
import org.apache.spark.ml.feature.VectorAssembler 
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
```
```ssh
import org.apache.spark.ml.feature.VectorIndexer

scala> import org.apache.spark.ml.feature.VectorAssembler 
import org.apache.spark.ml.feature.VectorAssembler

scala> import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.Pipeline

scala> import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession
```
	
##### 2. Opcional, utilice el siguiente codigo para quitar advertencias.

```scala
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```
```ssh
scala> import org.apache.log4j._
import org.apache.log4j._
```
 ##### 3. Inicia sesion de Spark

```scala
val spark = SparkSession.builder().getOrCreate()
```
```ssh
scala> val spark = SparkSession.builder().getOrCreate()
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@5af088bd
```

 ##### 4. Carga el archivo CSV

```scala
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Datos_Project_u3.csv")
```
```ssh
scala> val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Datos_Project_u3.cs
data: org.apache.spark.sql.DataFrame = [age: int, job: int ... 13 more fields]
```
 ##### 5. Muestra el nombre de las columnas

```scala
data.columns
```
```ssh
scala> data.columns
res6: Array[String] = Array(age, job, marital, education, default, balance, housing, loan, day, month, duration, campaign, pdays, previous)
```
##### 6. Imprime el esquema

```scala
data.printSchema()
```
```ssh
scala> data.printSchema()
root
 |-- age: integer (nullable = true)
 |-- job: integer (nullable = true)
 |-- marital: integer (nullable = true)
 |-- education: integer (nullable = true)
 |-- default: integer (nullable = true)
 |-- balance: integer (nullable = true)
 |-- housing: integer (nullable = true)
 |-- loan: integer (nullable = true)
 |-- day: integer (nullable = true)
 |-- month: integer (nullable = true)
 |-- duration: integer (nullable = true)
 |-- campaign: integer (nullable = true)
 |-- pdays: integer (nullable = true)
 |-- previous: integer (nullable = true)
 |-- y: integer (nullable = true)
```

##### 7. Muestra los primeros 5 renglones

```scala
data.head(5)
```
```ssh
scala> data.head(5)
res8: Array[org.apache.spark.sql.Row] = Array([58,0,0,0,0,2143,1,0,5,1,261,1,-1,0,0], [44,1,1,1,0,29,1,0,5,1,151,1,-1,0,0],1,-1,0,0], [47,3,0,4,0,1506,1,0,5,1,92,1,-1,0,0], [33,4,1,4,0,1,0,0,5,1,198,1,-1,0,0])
```

##### 8. Parametros estadisticos.

```scala
data.describe().show()
```
```ssh

scala> data.describe().show()
+-------+------------------+------------------+------------------+------------------+--------------------+-------------------------------+-----------------+------------------+-----------------+-----------------+------------------+----------------
|summary|               age|               job|           marital|         education|             default|           balanc
          loan|              day|             month|         duration|         campaign|             pdays|          previo
+-------+------------------+------------------+------------------+------------------+--------------------+-------------------------------+-----------------+------------------+-----------------+-----------------+------------------+----------------
|  count|             45211|             45211|             45211|             45211|               45211|             4521
         45211|            45211|             45211|            45211|            45211|             45211|             452
|   mean| 40.93621021432837| 3.526133020725045|0.6284090155050762|1.1320917475835526|0.018026586450200173|1362.27205768507622649355245405|15.80641879188693| 3.940722390568667|258.1630797814691|2.763840658246887| 40.19782796222158|0.58032337263055
| stddev|10.618762040975405|3.0983433930692805| 0.961539440002365|1.1191872671645307|  0.1330489390167441|3044.76582916852568200383232984|8.322476153044594|3.1797410024151826|257.5278122651706|3.098020883279192|100.12874599059828| 2.3034410449312
|    min|                18|                 0|                 0|                 0|                   0|             -801
             0|                1|                 1|                0|                1|                -1|                
|    max|                95|                11|                 3|                 4|                   1|            10212
             1|               31|                12|             4918|               63|               871|               2
+-------+------------------+------------------+------------------+------------------+--------------------+-------------------------------+-----------------+------------------+-----------------+-----------------+------------------+----------------
```
##### 9. Determina las columnas de entrada como features
```scala
val assembler = new VectorAssembler().setInputCols(Array("age","job","marital","education","default","balance","housing","loan","day","month","duration","campaign","pdays","previous")).setOutputCol("features")
```
```ssh
scala> val assembler = new VectorAssembler().setInputCols(Array("age","job","marital","education","default","balance","hous"duration","campaign","pdays","previous")).setOutputCol("features")
assembler: org.apache.spark.ml.feature.VectorAssembler = VectorAssembler: uid=vecAssembler_a00407bcd148, handleInvalid=error
```
##### 10. Features
```scala
val features = assembler.transform(data)
```
```ssh
scala> val features = assembler.transform(data)
features: org.apache.spark.sql.DataFrame = [age: int, job: int ... 14 more fields]
```

##### 11. Transforma los datos label categoricos a numericos
```scala
val indexerLabel = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(features)
```
```ssh
scala> val indexerLabel = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(features)
indexerLabel: org.apache.spark.ml.feature.StringIndexerModel = StringIndexerModel: uid=strIdx_c20ab1d154dc, handleInvalid=error
```
##### 12. Transforma los datos features categoricos a numericos con limitante de 4
```scala
val indexerFeatures = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(14)
```
```ssh
scala> val indexerFeatures = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(1
indexerFeatures: org.apache.spark.ml.feature.VectorIndexer = vecIdx_80b9574417b1
```
##### 13. Se crean los arreglos que seran usados para training y test 70% y 30%
```scala
val Array(training, test) = features.randomSplit(Array(0.7, 0.3), seed = 12345)
```
```ssh
scala> val Array(training, test) = features.randomSplit(Array(0.7, 0.3), seed = 12345)
training: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [age: int, job: int ... 14 more fields]
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [age: int, job: int ... 14 more fields]
```
##### 14. Parametros del algoritmo (Entradas, capa 1, capa 2, 2 salidas)
```scala
val Array(training, test) = features.randomSplit(Array(0.7, 0.3), seed = 12345)
```
##### 15. Estructura del modelo Multilayer Perceptron
```scala
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(1234).setMaxIter(100)
```
```ssh
scala> val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("inde(128).setSeed(1234).setMaxIter(100)
trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_aa26211705cd
```
##### 16. Convierte de numerico a categoricos los datos "Prediction"
```scala
val converterLabel = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(indexerLabel.labels)
```

```ssh
scala> val converterLabel = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(indexerL
warning: one deprecation (since 3.0.0); for details, enable `:setting -deprecation' or `:replay -deprecation'
converterLabel: org.apache.spark.ml.feature.IndexToString = idxToStr_1e83e81d39c1
```
##### 17. Ingreso de datos al Pipeline
```scala
val pipeline = new Pipeline().setStages(Array(indexerLabel, indexerFeatures, trainer, converterLabel))
```
```ssh
scala> val pipeline = new Pipeline().setStages(Array(indexerLabel, indexerFeatures, trainer, converterLabel))
pipeline: org.apache.spark.ml.Pipeline = pipeline_4dbaed6c8bd8
```
##### 18. Entrena el modelo
```scala
val model = pipeline.fit(training)
```
```ssh
scala> val model = pipeline.fit(training)
model: org.apache.spark.ml.PipelineModel = pipeline_2f80d60f9c25
```
##### 19. Evalua el modelo
```scala
val results = model.transform(test)
```
```ssh
scala> val results = model.transform(test)
results: org.apache.spark.sql.DataFrame = [age: int, job: int ... 20 more fields]
```
##### 20. Resultados
```scala
results.show()
```
```ssh
+---+---+-------+---------+-------+-------+-------+----+---+-----+--------+--------+-----+--------+---+--------------------+------------+--------------------+--------------------+--------------------+----------+--------------+
|age|job|marital|education|default|balance|housing|loan|day|month|duration|campaign|pdays|previous|  y|            features|indexedLabel|     indexedFeatures|       rawPrediction|         probability|prediction|predictedLabel|
+---+---+-------+---------+-------+-------+-------+----+---+-----+--------+--------+-----+--------+---+--------------------+------------+--------------------+--------------------+--------------------+----------+--------------+
| 18| 11|      1|        4|      0|     35|      0|   0| 21|    4|     104|       2|   -1|       0|  0|[18.0,11.0,1.0,4....|         0.0|[18.0,11.0,1.0,3....|[1.23142230690192...|[0.94114425680076...|       0.0|             0|
| 19| 11|      1|        1|      0|     55|      0|   0|  6|    5|      89|       2|  193|       1|  0|[19.0,11.0,1.0,1....|         0.0|[19.0,11.0,1.0,1....|[0.38851297146708...|[0.75886394216340...|       0.0|             0|
| 19| 11|      1|        1|      0|    302|      0|   0| 16|    3|     205|       1|   -1|       0|  1|[19.0,11.0,1.0,1....|         1.0|[19.0,11.0,1.0,1....|[0.68986081574383...|[0.84651537088173...|       0.0|             0|
| 19| 11|      1|        1|      0|    526|      0|   0| 14|    7|     122|       3|   -1|       0|  0|[19.0,11.0,1.0,1....|         0.0|[19.0,11.0,1.0,1....|[1.33576480659663...|[0.94005436374586...|       0.0|             0|
| 19| 11|      1|        1|      0|    527|      0|   0|  4|   12|     154|       3|   -1|       0|  0|[19.0,11.0,1.0,1....|         0.0|[19.0,11.0,1.0,1....|[0.24517430177422...|[0.72853029168218...|       0.0|             0|
| 19| 11|      1|        3|      0|      0|      0|   0|  4|    1|      72|       4|   -1|       0|  0|[19.0,11.0,1.0,3....|         0.0|[19.0,11.0,1.0,2....|[1.23386498337818...|[0.94132417332970...|       0.0|             0|
| 19| 11|      1|        3|      0|    608|      0|   0| 12|    1|     236|       1|  180|       2|  1|[19.0,11.0,1.0,3....|         1.0|[19.0,11.0,1.0,2....|[0.24517173109900...|[0.72852946720153...|       0.0|             0|
| 20|  3|      0|        3|      0|   -172|      1|   1| 19|    1|     238|       3|   -1|       0|  0|[20.0,3.0,0.0,3.0...|         0.0|[20.0,3.0,0.0,2.0...|[1.84566275589565...|[0.97305623514879...|       0.0|             0|
| 20|  3|      1|        1|      0|    423|      1|   0| 16|   11|     498|       1|   -1|       0|  1|[20.0,3.0,1.0,1.0...|         1.0|[20.0,3.0,1.0,1.0...|[0.24517156861184...|[0.72852942301886...|       0.0|             0|
| 20| 11|      1|        1|      0|    162|      0|   0| 25|    1|     106|       2|   -1|       0|  0|[20.0,11.0,1.0,1....|         0.0|[20.0,11.0,1.0,1....|[1.34521169750010...|[0.94090948987273...|       0.0|             0|
| 20| 11|      1|        1|      0|    291|      0|   0| 11|    1|     172|       5|  371|       5|  0|[20.0,11.0,1.0,1....|         0.0|[20.0,11.0,1.0,1....|[0.24517173109900...|[0.72852946720153...|       0.0|             0|
| 20| 11|      1|        3|      0|      0|      0|   0|  1|    9|     143|       5|   91|       8|  0|[20.0,11.0,1.0,3....|         0.0|[20.0,11.0,1.0,2....|[0.39935379205912...|[0.78278627855213...|       0.0|             0|
| 20| 11|      1|        4|      0|    179|      0|   0| 25|   10|     196|       3|  182|       3|  0|[20.0,11.0,1.0,4....|         0.0|[20.0,11.0,1.0,3....|[0.24517249409483...|[0.72852971191378...|       0.0|             0|
| 20| 11|      1|        4|      0|    479|      0|   0| 11|   12|     158|       2|   -1|       0|  0|[20.0,11.0,1.0,4....|         0.0|[20.0,11.0,1.0,3....|[0.26357112768511...|[0.73438983740847...|       0.0|             0|
| 20| 11|      1|        4|      0|    829|      0|   0|  9|    3|     253|       2|   -1|       0|  1|[20.0,11.0,1.0,4....|         1.0|[20.0,11.0,1.0,3....|[0.24517173109900...|[0.72852946720153...|       0.0|             0|
| 20| 11|      1|        4|      0|   4137|      1|   0| 16|   11|      55|       2|   -1|       0|  0|[20.0,11.0,1.0,4....|         0.0|[20.0,11.0,1.0,3....|[0.99186208352942...|[0.90441382303000...|       0.0|             0|
| 21|  0|      1|        0|      0|    243|      0|   1| 17|    2|     181|       2|   -1|       0|  0|[21.0,0.0,1.0,0.0...|         0.0|[21.0,0.0,1.0,0.0...|[1.34520366345050...|[0.94090876751990...|       0.0|             0|
| 21|  0|      1|        1|      0|    691|      0|   0| 13|    4|     219|       1|  101|       2|  0|[21.0,0.0,1.0,1.0...|         0.0|[21.0,0.0,1.0,1.0...|[0.24517173109916...|[0.72852946720158...|       0.0|             0|
| 21|  3|      1|        1|      0|    614|      1|   0| 13|    1|     243|       3|   -1|       0|  0|[21.0,3.0,1.0,1.0...|         0.0|[21.0,3.0,1.0,1.0...|[0.24517175891996...|[0.72852947612442...|       0.0|             0|
| 21|  3|      1|        1|      0|    820|      1|   0| 27|    2|     302|       2|   -1|       0|  0|[21.0,3.0,1.0,1.0...|         0.0|[21.0,3.0,1.0,1.0...|[0.24517173110501...|[0.72852946720346...|       0.0|             0|
+---+---+-------+---------+-------+-------+-------+----+---+-----+--------+--------+-----+--------+---+--------------------+------------+--------------------+--------------------+--------------------+----------+--------------+
only showing top 20 rows

```
##### 21. Metricas a evaluar
```scala
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
```
```ssh
scala> val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = MulticlassClassificationEvaluator: uid=mcEval_b7c672c42e7f, metricName=accuracy, metricLabel=0.0, beta=1.0, eps=1.0E-15
```
##### 22. Resultados
```scala
val accuracy = evaluator.evaluate(results)

```
```ssh
scala> val accuracy = evaluator.evaluate(results)
accuracy: Double = 0.882258064516129
```
