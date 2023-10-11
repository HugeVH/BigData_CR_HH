//Fib_  v1
//#Código Fibbonaci 

//## Algoritmo 1

var i = 7

def fib(i:Int): Int =
 {
   if (i<2)
    return i
   else 
    return fib(i-1)+fib(i-2)
    
 }
 
 println("El numero Fibonacci de " + i + " es: " + fib(i))

