# Nuestra branch unit2 
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

scala> var iris = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")
iris: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 3 more fields]

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


scala> iris.printSchema()
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)


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


scala> val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width")).setOutputCol("features")   
assembler: org.apache.spark.ml.feature.VectorAssembler = VectorAssembler: uid=vecAssembler_fc2de4a451a0, handleInvalid=error, numInputCols=4

scala> val features = assembler.transform(iris)
features: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 4 more fields]

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