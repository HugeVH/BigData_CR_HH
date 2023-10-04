//4. Crea un arreglo de numero en rango del 1-1000 en pasos de 5 en 5
val arr = Array.range(1,1000,5)

//5. Cuales son los elementos únicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversión a conjuntos

val Lista = List (1,3,3,4,6,7,3,7)
val unicosLista = Lista.distinct

//Crea una mapa mutable llamado nombres que contenga los siguiente
var nombres = Map(("Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana", "27"))

//6 a . Imprime todas la llaves del mapa
nombres.keys
println(s"Todas llaves $nombres.keys")

