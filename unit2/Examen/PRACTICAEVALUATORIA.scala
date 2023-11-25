import org.apache.spark.ml.classification.MultilayerPerceptronClassifier 
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.IndexToString 
import org.apache.spark.ml.feature.StringIndexer 
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline

var iris = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")

//data.columns()
iris.show() 
iris.printSchema()
iris.show(5) 

iris.describe().show() 

val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width")).setOutputCol("features")
//Total las caracteristicas
val features = assembler.transform(iris)
//Agarra un dato categorico y lo vuelve numerico
val indexerLabel = new StringIndexer().setInputCol("species").setOutputCol("indexedLabel").fit(features)

val indexerFeatures = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)

val Array(training, test) = features.randomSplit(Array(0.7, 0.3), seed = 12345) 
//La semilla es para separar

val layers = Array[Int](4, 5, 4, 3) //neurona input,layers oculto 1, layer oculto 2, numero de salidas (3) son las especies cetosa o sea 3 neuronas de salida
// los datos ubicados en la posicion 5 y 4 se pueden cambiar
//features entradas--label salidas

//Se configura la estructura del modelo de tipo multilayer perceptron
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(1234).setMaxIter(100) 

//convierte de numerico a categorico (string) los datos de prediction
val converterLabel = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(indexerLabel.labels)

//Mete los datos al contenedor (tuberia)
val pipeline = new Pipeline().setStages(Array(indexerLabel, indexerFeatures, trainer, converterLabel))
//se entrena el modelo
val model = pipeline.fit(training)

val results = model.transform(test)

//Evalua el modelo con las metricas
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

val accuracy = evaluator.evaluate(results)

println("Error = " + (1.0 - accuracy))

//results.show()