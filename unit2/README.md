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