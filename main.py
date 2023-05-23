import random
from typing import Callable, List
from functools import reduce, partial
import pandas as pd

import matplotlib.pyplot as plt

#Parametros para variar entre corridas
CANTIDAD_ELITISMO = 2
CROSSOVER_PROB = 0.75
MUTACION_PROB = 0.05

CANTIDAD_POBLACION = 10
CANTIDAD_CICLOS = 100
CANTIDAD_GENES = 30

"""
COMIENZO DEL PROGRAMA
"""

def menu_elitismo():
    print("ELITISMO")
    print()
    print("E- Para usar elitismo")
    print("N- No usar elitismo")

def menu_seleccion():
    print("SELECCION")
    print()
    print("T- Seleccion por torneo")
    print("R- Seleccion por ruleta")

# Funcion que pasa de binario a decimal.
def binario_a_decimal(cromosoma):
    """Funcion que pasa un cromosoma de binario a decimal"""
    decimal = 0
    posicion = 0
    for digito in reversed(cromosoma):
        decimal += int(digito) * 2 ** posicion
        posicion += 1

    return decimal


def funcion_objetivo(cromosoma):
    """Funcion que recibe un cromosoma,
    lo pasa a decimal y retorna la funcion objetivo"""
    decimal = binario_a_decimal(cromosoma)
    return (decimal / ((2 ** CANTIDAD_GENES) - 1)) ** 2


def fitness(total, poblacion_f_obj):
    """
    Funcion que evalua el fitness de toda la poblacion dada
    una lista cuyos elementos son la funcion objetivo evaluada en
    cada cromosoma
    """
    poblacion_fitness = [num_obj / total for num_obj in poblacion_f_obj]
    return poblacion_fitness


def generar_ruleta(poblacion_fitness):
    """Funcion que recibe el fitness de cada elemento de la poblacion
    y genera una ruleta de 1000 posiciones, cada posicion contiene
    la posicion que ocupa su respectivo cromosoma en la lista poblacion"""
    ruleta = []
    comienzo = 0
    for i, num_fitness in enumerate(poblacion_fitness):
        peso_en_ruleta = int(round(num_fitness, 3) * 1000)
        for j in range(comienzo, peso_en_ruleta + comienzo):
            ruleta.append(i)
        comienzo += peso_en_ruleta + 1
    return ruleta


def seleccion_ruleta(poblacion, poblacion_fitness):
    """Funcion que dada una poblacion y una ruleta, selecciona los
    2 padres mas aptos para realizar el crossover"""
    # Generamos la ruleta con la funcion generar_ruleta
    ruleta = generar_ruleta(poblacion_fitness)
    MAXIMO = len(ruleta) - 1
    #Elegimos la posicion de la ruleta
    indice_1 = ruleta[random.randint(0, MAXIMO)]
    indice_2 = ruleta[random.randint(0, MAXIMO)]

    #Buscamos el elemento de la poblacion al que apunta lo que seleccionamos en la ruleta
    cromo_1 = poblacion[indice_1]
    cromo_2 = poblacion[indice_2]
    return cromo_1, cromo_2

def seleccion_torneo(poblacion, poblacion_fitness):
    seleccionados = []
    for i in range(2):

        #torneo = random.sample(poblacion, 2) # selecciona 2 individuos al azar de la población
        """
        aptitudes = [individuo.aptitud() for individuo in torneo] # evalúa la aptitud de cada uno
        """
        torneo = [random.randint(0,len(poblacion)-1), random.randint(0,len(poblacion)-1)]
        max_poblacion_fitness=max([poblacion_fitness[i] for i in torneo])
        index_max_fitness = poblacion_fitness.index(max_poblacion_fitness)
        seleccionados.append(poblacion[index_max_fitness])
    return tuple(seleccionados)

def crossover(padre_1:[], padre_2:[]):
    """
    Funcion que dados 2 cromosomas padres, realiza el crossover y retorna
    los hijos
    """
    if random.random() < CROSSOVER_PROB:
        # Seleccionamos el punto de corte
        corte = random.randint(0, CANTIDAD_GENES - 1)

        # Cortamos los padres en inicio y fin y creamos los hijos
        inicio_hijo_1 = padre_1[0:corte]
        inicio_hijo_2 = padre_2[0:corte]

        fin_hijo_1 = padre_2[corte:CANTIDAD_GENES]
        fin_hijo_2 = padre_1[corte:CANTIDAD_GENES]

        hijo_1 = inicio_hijo_1 + fin_hijo_1
        hijo_2 = inicio_hijo_2 + fin_hijo_2

        return hijo_1, hijo_2
    else:
        return padre_1, padre_2


def mutacion(cromo):
    """funcion que muta el gen que se le pasa como parametro"""
    if random.random() < MUTACION_PROB:
        #Seleccionamos el gen a mutar
        gen = random.randint(0, CANTIDAD_GENES - 1)

        #Obtenemos el gen opuesto
        gen_mutado = int(not cromo[gen])

        #Asignamos el gen opuesto al cromosoma
        cromo[gen] = gen_mutado



def generar_poblacion_simple(poblacion,cantidad_elementos, seleccion:Callable):
    """Funcion que genera una poblacion de manera simple sin elitismo, se usa para llamar en generar_poblacion"""
    # Evaluamos el fitness de toda la poblacion
    poblacion_fitness = fitness(acum, poblacion_f_obj)

    # Generamos la poblacion
    poblacion_nueva = []
    for j in range(cantidad_elementos // 2):

        #Elegimos los padres
        padre_1, padre_2 = seleccion(poblacion, poblacion_fitness)

        #Hacemos el crossover y mutamos los hijos
        hijo_1, hijo_2 = crossover(padre_1, padre_2)
        mutacion(hijo_1)
        mutacion(hijo_2)

        #Agregamos los hijos a la poblacion
        poblacion_nueva.append(hijo_1)
        poblacion_nueva.append(hijo_2)

    return poblacion_nueva

def generar_poblacion(poblacion:[],cantidad_elementos:int,seleccion:Callable,  cantidad_elitismo:int =0):
    """Funcion que se utiliza para generar poblacion, recibe una poblacion inicia, una cantidad de elementos para generar la poblacion
     um metodo de seleccion y una cantidad de elementos con elitismo
     """
    def elitismo(poblacion,cantidad_elitismo):
        """Funcion que dada una poblacion y su fitness, selecciona los 2 cromosomas con mayor fitness"""
        poblacion_ordenada = sorted(poblacion, key=funcion_objetivo, reverse=True)

        return [poblacion_ordenada[i] for i in range(cantidad_elitismo)]
    poblacion_nueva = []
    if(cantidad_elitismo > 0):
        poblacion_nueva += elitismo(poblacion, cantidad_elitismo)
    poblacion_nueva += generar_poblacion_simple(poblacion,cantidad_elementos- cantidad_elitismo, seleccion )
    return poblacion_nueva
def buscar_max_cromo(poblacion):
    """Funcion que busca el cromosoma maximo """
    max = -1
    max_cromo = poblacion[0]

    for  cromo in poblacion:
        cromo_f_obj = funcion_objetivo(cromo)
        if cromo_f_obj > max:
            max_cromo = cromo
            max = cromo_f_obj

    return max_cromo


def buscar_min_cromo(poblacion):
    """Funcion que busca el cromosoma minimo"""
    min = 1
    min_cromo = poblacion[0]

    for cromo in poblacion:
        cromo_f_obj = funcion_objetivo(cromo)
        if cromo_f_obj < min:
            min_cromo = cromo
            min = cromo_f_obj

    return min_cromo




def cromo_to_string(cromosoma):
    """Funcion que dado un cromosoma como una lista, retorna un numero binario como string"""
    str_cromo = ""
    for bit in cromosoma:
        str_cromo += str(bit)

    return str_cromo

poblacion = []
cromosoma = []
poblacion_f_obj = []
poblacion_fitness = []
ruleta = []
acum = 0

maximo_historico = []
minimo_historico = []
promedio_historico = []
cromosoma_maximo_historico = []
cromosoma_minimo_historico = []

poblacion_inicial = []


"""
    INICIO DEL PROGRAMA
"""

#
print("INICIO DE PROGRAMA")
menu_elitismo()
tipo_elitismo = input().upper()
teclas_posibles_elitismo = ("E", "N")
while tipo_elitismo not in teclas_posibles_elitismo:
    print("No escogio tecla valida, vuelva a ingresar")
    menu_elitismo()
    tipo_elitismo = input().upper()
cant_elitismo = 0
if tipo_elitismo == "E":
   cant_elitismo = CANTIDAD_ELITISMO




menu_seleccion()
tipo_seleccion = input().upper()
teclas_posibles_seleccion = ("R", "T")
while tipo_seleccion not in teclas_posibles_seleccion:
    print("No escogio tecla valida, seleccione tecla valida")
    menu_seleccion()
    tipo_seleccion = input().upper()



# GENERAMOS POBLACION INICIAL ALEATORIA
# Y calculamos el minimo, el maximo y el promedio
for i in range(CANTIDAD_POBLACION):
    for j in range(CANTIDAD_GENES):
        cromosoma.append(random.randint(0, 1))
    poblacion_inicial.append(cromosoma)
    cromosoma = []

poblacion_f_obj = [funcion_objetivo(cromo) for cromo in poblacion_inicial]
acum = sum(poblacion_f_obj)
maximo_cromo_inic = buscar_max_cromo(poblacion_inicial)
minimo_cromo_inic = buscar_min_cromo(poblacion_inicial)

maximo_inic = max(poblacion_f_obj)
cromosoma_maximo_historico.append(cromo_to_string(maximo_cromo_inic))
cromosoma_minimo_historico.append(cromo_to_string(minimo_cromo_inic))

minimo_inic = min(poblacion_f_obj)
promedio_inic = sum(poblacion_f_obj) / CANTIDAD_POBLACION

maximo_historico.append(maximo_inic)
minimo_historico.append(minimo_inic)
promedio_historico.append(promedio_inic)


poblacion = poblacion_inicial
for i in range(1, CANTIDAD_CICLOS):

    # Evaluamos el fitness de toda la poblacion
    poblacion_fitness = fitness(acum, poblacion_f_obj)

    # Generamos la poblacion con elitismo
    if tipo_seleccion == "R":
        poblacion = generar_poblacion(poblacion,CANTIDAD_POBLACION,seleccion_ruleta, cant_elitismo)
    elif tipo_seleccion == "T":
        poblacion = generar_poblacion(poblacion, CANTIDAD_POBLACION, seleccion_torneo, cant_elitismo)

    # APLICAMOS FUNCION OBJETIVO A CADA ELEMENTO DE LA POBLACION
    poblacion_f_obj = [funcion_objetivo(cromo) for cromo in poblacion]

    # OBTENEMOS EL TOTAL
    acum = sum(poblacion_f_obj)
    maximo = max(poblacion_f_obj)
    minimo = min(poblacion_f_obj)
    promedio = acum / CANTIDAD_POBLACION
    maximo_cromo = buscar_max_cromo(poblacion)
    minimo_cromo = buscar_min_cromo(poblacion)

    cromosoma_maximo_historico.append(cromo_to_string(maximo_cromo))
    cromosoma_minimo_historico.append(cromo_to_string(minimo_cromo))

    maximo_historico.append(maximo)
    minimo_historico.append(minimo)
    promedio_historico.append(promedio)




##Comparamos los resultados de la poblacion inicial con la poblacion final
print("Poblacion Inicial")

print(f"maximo: {maximo_inic} --- minimo: {minimo_inic} --- promedio: {promedio_inic}")
print(f"maximo_cromo: {maximo_cromo_inic} --- minimo_cromo: {minimo_cromo_inic}\n\n")
print("Poblacion final")
print(f"maximo: {maximo} --- minimo: {minimo} --- promedio: {promedio}")
print(f"maximo_cromo:{maximo_cromo} --- minimo_cromo:{minimo_cromo}")

poblaciones = list(range(CANTIDAD_CICLOS))


#Mostrar graficos historico
plt.plot(poblaciones,maximo_historico,"g", label = "max")
plt.plot(poblaciones,minimo_historico, "b", label = "avg")
plt.plot(poblaciones,promedio_historico, "r", label = "min")
plt.title("Máximo, mínimo y promedio Historico")
plt.xlabel("Numero de iteraciones")
plt.ylabel("Valores")
plt.show()


print("Presione E si desea generar un excel")
pasa_a_excel  = input().upper()
if pasa_a_excel == "E":
    data = {"nro_poblacion":poblaciones, "maximo":maximo_historico, "minimo": minimo_historico,
            "promedio": promedio_historico, "maximo_cromo": cromosoma_maximo_historico,
            "minimo_cromo": cromosoma_minimo_historico}

    df = pd.DataFrame(data, columns=["nro_poblacion", "maximo", "minimo", "promedio","maximo_cromo", "minimo_cromo"])

    df.to_excel("algoritmo_genetico.xlsx", header = True)

