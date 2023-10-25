# Primer examen parcial de Big Data

##### 6.- Utiliza el metodo Describe() para aprender mas sobre el DataFrame()
- El metodo Describe es utilizado para mostrar parametros estadisticos para cada una de las columnas de datos.
	a) El primer parametro es count, el cual muestra la cantidad de datos que se tienen.
	b) El segundo parametro es mean es la media de datos, es una medida de tendencia central al igual que la mediana y moda.
	c) El tercer parametro es stddev, la cual es una medida de dispersion que indica que tan alejados se encuentran los datos respecto a la media.
	d) los siguientes parametros (min y max) definen el maximo y minimo de los datos por columna. Consecuentemente, permiten obtener el rango de datos.

##### 7.- Crea un nuevo Dataframe a partir del df creado llamado "HVRatio" que es la relacion que existe entre el precio de la 
##### columna High frente a la columna "Volumen" de acciones negociadas por un dia.

a) Esta relacion es para saber cuantas acciones se vendieron en un dia de trabajo.

##### 8.-Que dia tuvo el pico mas alto la columna Open??
a) R. dia 708

##### 9. ¿Cuál es el significado de la columna Cerrar “Close” en el contexto de información
##### financiera, explíquelo no hay que codificar nada?
R. El valor de la accion al momento de cerrar el dia.

##### 10. ¿Cuál es el máximo y mínimo de la columna “Volumen”?
R. min= 3531300, max= 315541800

##### 11. Con Sintaxis Scala/Spark $ conteste lo siguiente:
##### a) ¿Cuántos días fue la columna “Close” inferior a $ 600?
R. 1218.

##### ¿Qué porcentaje del tiempo fue la columna “High” mayor que $ 500?
R. 4.29

##### c) ¿Cuál es la correlación de Pearson entre columna “High” y la columna
##### “Volumen”?
R. -0.209 "una correlacion debil"

##### d) ¿Cuál es el máximo de la columna “High” por año?
2015---716.15
2013---389.15
2014---489.29
2012---133.42
2016---129.28
2011---120.28

##### e) ¿Cuál es el promedio de la columna “Close” para cada mes del calendario?
Mes----Close
12-----199.37
1---212.22
6---295.15
3---249.58
5---264.37
9---206.09
4---246.97
8---195.25
7---243.64
10---205.93
11---194.31
2---254.19







### Analisis del codigo de la sesion 6

- Analice y describa cada unas de las funciones del codigo en la sesion 6 del tema Spark-Basics y finalmente documenta en el archivo readme.md
en el repositorio correspondiente.

### Ejercicio 1
Este codigo muestra dos listas (arreglos), de los cuales analiza uno de ellos. El objetivo de este algoritmo es mostrar si un numero es divisible entre dos o no, para lo cual añade las etiquetas: Si el residuo de la division es cero se añade la etiqueta "even", en caso contrario la etiqueta "odd".

### Ejercicio 2
En el siguiente ejercicio, se muestra un algoritmo el cual desplega tres listas, de las cuales menciona si los numeros son mayor a 7, suma 14, caso contrario solo suma el valor ingresado.

###Ejercicio 3 
es una funcion que te regresa el string invertido
