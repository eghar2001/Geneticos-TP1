import random
from functools import reduce

# print(random.random())
# asdasdasdas

CROSSOVER_PROB = 0.75
MUTACION_PROB = 0.05

CANTIDAD_POBLACION = 10
CANTIDAD_CICLOS = 100
CANTIDAD_GENES = 30
CANTIDAD_CORTES = 1


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


def seleccion(poblacion, ruleta):
    """Funcion que dada una poblacion y una ruleta, selecciona los
    2 padres mas aptos para realizar el crossover"""
    MAXIMO = len(ruleta) - 1
    #Elegimos la posicion de la ruleta
    indice_1 = ruleta[random.randint(0, MAXIMO)]
    indice_2 = ruleta[random.randint(0, MAXIMO)]

    #Buscamos el elemento de la poblacion al que apunta lo que seleccionamos en la ruleta
    cromo_1 = poblacion[indice_1]
    cromo_2 = poblacion[indice_2]
    return cromo_1, cromo_2


def crossover(padre_1, padre_2):
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


poblacion = []
cromosoma = []
poblacion_f_obj = []
poblacion_fitness = []
ruleta = []
acum = 0

poblacion_inicial = []
# GENERAMOS POBLACION INICIAL ALEATORIA
# Y calculamos el minimo, el maximo y el promedio
for i in range(CANTIDAD_POBLACION):
    for j in range(CANTIDAD_GENES):
        cromosoma.append(random.randint(0, 1))
    poblacion_inicial.append(cromosoma)
    cromosoma = []

poblacion_f_obj = [funcion_objetivo(cromo) for cromo in poblacion_inicial]
acum = sum(poblacion_f_obj)

maximo_inic = max(poblacion_f_obj)
minimo_inic = min(poblacion_f_obj)
promedio_inic = sum(poblacion_f_obj) / CANTIDAD_POBLACION
print("Poblacion Inicial")
print(f"{maximo_inic=} --- {minimo_inic=} --- {promedio_inic=}\n\n")

poblacion = poblacion_inicial
for i in range(CANTIDAD_CICLOS):

    # Evaluamos el fitness de toda la poblacion
    poblacion_fitness = fitness(acum, poblacion_f_obj)

    # Generamos la poblacion
    poblacion_nueva = []
    for i in range(1, CANTIDAD_POBLACION // 2):
        #Generamos la ruleta con la funcion generar_ruleta
        ruleta = generar_ruleta(poblacion_fitness)

        #Elegimos los padres
        padre_1, padre_2 = seleccion(poblacion, ruleta)

        #Hacemos el crossover y mutamos los hijos
        hijo_1, hijo_2 = crossover(padre_1, padre_2)
        mutacion(hijo_1)
        mutacion(hijo_2)

        #Agregamos los hijos a la poblacion
        poblacion_nueva.append(hijo_1)
        poblacion_nueva.append(hijo_2)
    #Seteamos la poblacon
    poblacion = poblacion_nueva

    # APLICAMOS FUNCION OBJETIVO A CADA ELEMENTO DE LA POBLACION
    poblacion_f_obj = [funcion_objetivo(cromo) for cromo in poblacion]

    # OBTENEMOS EL TOTAL
    acum = sum(poblacion_f_obj)
    maximo = max(poblacion_f_obj)
    minimo = min(poblacion_f_obj)
    promedio = acum / CANTIDAD_POBLACION
    print(f"CORRIDA: {i}\n")
    print(f"{maximo=} --- {minimo=} --- {promedio=}\n\n")

##Comparamos los resultados de la poblacion inicial con la poblacion final
print("Poblacion Inicial")

print(f"{maximo_inic=} --- {minimo_inic=} --- {promedio_inic=}\n\n")

print("Poblacion final")
print(f"{maximo=} --- {minimo=} --- {promedio=}\n\n")

