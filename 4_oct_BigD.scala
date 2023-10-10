import util.control.Breaks._
//se declara una variable tipo int
var x = 0
//condicion while, mientras x se manor a 5
while(x < 5){
    //se imprime el valor actual de x
    println(s"x is currently $x")
    // se imprime la etiqueta:
    println(s"x is still less then 5, adding 1 to x ")
    //se incrementa en uno el valor de la variable x
    x = x + 1
}

//Declara variable y=0
var y = 0
//Mientras y<10, se imprimen dos mensajes y se pone condicion
while(y < 10){
    println(s"y is currently $y")
    println(s"y is still less then 10, add 1 to y")
    //incrementa el valor de la variable en uno
    y = y+1
    //compara el valor de la variable con el numero 3, es decir compara caracteres
    if(y==3) break 

}
println("###########")


// Functions 
//Define la funcion "simple" sin argumentos y no devuelve valores "unit"
def simple(): Unit = {
    //imprime la etiqueta print
    println("simple print")
}
//llamada a la funcion simple
simple()

//Funcion añadir la cual tiene dos parametro de entrada tipo entero_Hasta aqui inicio Carlos.
def adder(num1:Int, num2:Int): Int = {
    //regresa la suma de dos valores
    return num1 + num2
}

//Esta es la llamada a la funcion_la cua lleva por nombre adder
adder(5, 5)
//Define una variable sin retorno name
def greetName(name:String): String{
    return s"Hello $name"
}

//Funcion greetName tipo String y regresa un string
def greetName(name:String): String={
    //Concatenacion de las cadenas
    return s"Hello $name"
}
// declara "fullgreat" y le asigna el valor devuelto por la fucion greetname
val fullgreet = greetName("Christian")
//imprime el valor de la funcion greetname
println(fullgreet)


//Funcion que evalua si un numero es primo
def isPrime(num:Int): Boolean = {
    for(n <- Range(2, num)){
        //Condicion si la division es cero
        if(num%n == 0){
            return false
        }
    }
    return true
}
println(isPrime(10))
println(isPrime(23))

//se declara una lista con numeros enteros
val numbers = List(1,2,3,7)

//Se declara la funcion check la cual recibe una lista de enteros y regresa una lista de enteros
def check(nums:List[Int]): List[Int]={
    return nums
}

println(check(numbers))