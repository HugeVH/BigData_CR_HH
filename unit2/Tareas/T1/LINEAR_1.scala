

////////////////////////////////////////////
//// LINEAR REGRESSION EXERCISE ///////////
/// Coplete las tareas comentadas ///
/////////////////////////////////////////

// Import LinearRegression
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

// Opcional: Utilice el siguiente codigo para configurar errores
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


// Inicie una simple Sesion Spark
val spark = SparkSession.builder().getOrCreate()

// Utilice Spark para el archivo csv Clean-Ecommerce .
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")

// Imprima el schema en el DataFrame. //imprime las columnas del archivo csv
data.printSchema()

// Imprima un renglon de ejemplo del DataFrane.
data.head(1)



//////////////////////////////////////////////////////
//// Configure el DataFrame para Machine Learning ////
//////////////////////////////////////////////////////

// Transforme el data frame para que tome la forma de
// ("label","features")

// Importe VectorAssembler y Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
// Renombre la columna Yearly Amount Spent como "label"
// Tambien de los datos tome solo la columa numerica 
// Deje todo esto como un nuevo DataFrame que se llame df
data.columns
val df = data.select(data("Yearly Amount Spent").as("label"), $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership")

// Que el objeto assembler convierta los valores de entrada a un vector


// Utilice el objeto VectorAssembler para convertir la columnas de entradas del df
// a una sola columna de salida de un arreglo llamado  "features"
// Configure las columnas de entrada de donde se supone que leemos los valores.
// Llamar a esto nuevo assambler

val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership")).setOutputCol("features")

// Utilice el assembler para transform nuestro DataFrame a dos columnas: label and features
val output = assembler.transform(df).select($"label", $"features")


// Crear un objeto para modelo de regresion linea.
var lr = new LinearRegression()

// Ajuste el modelo para los datos y llame a este modelo lrModelo

var lrModelo = lr.fit(output)


// Imprima the  coefficients y intercept para la regresion lineal
lrModelo.coefficients
lrModelo.intercept
// Resuma el modelo sobre el conjunto de entrenamiento imprima la salida de algunas metricas!
// Utilize metodo .summary de nuestro  modelo para crear un objeto
// llamado trainingSummary
// Muestre los valores de residuals, el RMSE, el MSE, y tambien el R^2 .
val trainingSummary = lrModelo.summary

trainingSummary.residuals.show()
trainingSummary.rootMeanSquaredError
trainingSummary.meanSquaredError
trainingSummary.r2

// Buen trabajo!