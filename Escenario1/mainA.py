import numpy as np  # Manejar matrices
import gzip         # Descomprimir archivos del dataset
import os           # Administrador de archivos
import time         # Medir tiempos

#region Constantes

# Constante tamaño de la imagen 28x28 píxeles
TAMAÑO_ENTRADA = 784 

# Constante tamaño de la capa oculta
TAMAÑO_CAPA_OCULTA = 512

# Constante tamaño de la salida (10 dígitos)
TAMAÑO_SALIDA = 10

#endregion

#region Configuración Entrenamiento

# Tasa de aprendizaje
TASA_APRENDIZAJE = 0.01

# Número de épocas (Cuántas veces mostrar a la red todo el dataset)
EPOCAS = 10

# Tamaño del batch (Mostrar 64 imágenes para calcular error y corregir)
TAMAÑO_BATCH = 64

#endregion

#region Carga datos

# Función para cargar imágenes del dataset MNIST y convertirlas a formato adecuado
def cargar_imagenes_dataset(nombre_archivo):
    with gzip.open(nombre_archivo, 'rb') as f:
        # Ignorar encabezado propio de MNIST
        # Leer datos como un array de numpy de enteros sin signo de 8 bits
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # Transformación de datos de una tira a una matriz de (cantidad_imágenes, tamaño_entrada)
    # División por 255 para normalizar valores entre 0 y 1
    return data.reshape(-1, TAMAÑO_ENTRADA).astype(np.float32) / 255.0

def cargar_etiquetas_dataset(nombre_archivo):
    with gzip.open(nombre_archivo, 'rb') as f:
        # Ignorar encabezado propio de MNIST
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

print("Cargando dataset MNIST...")
X_train = cargar_imagenes_dataset(os.path.join('../Resources', 'train-images-idx3-ubyte.gz'))
y_train = cargar_etiquetas_dataset(os.path.join('../Resources', 'train-labels-idx1-ubyte.gz'))
X_test = cargar_imagenes_dataset(os.path.join('../Resources', 't10k-images-idx3-ubyte.gz'))
y_test = cargar_etiquetas_dataset(os.path.join('../Resources', 't10k-labels-idx1-ubyte.gz'))

print(f"Dataset cargado con {X_train.shape[0]} imágenes de entrenamiento y {X_test.shape[0]} imágenes de prueba.")

#endregion

#region Inicialización

# Semilla para que siempre genere los mismos números aleatorios
np.random.seed(42)

def inicializar_parametros():
    # CAPA 1: Entrada a Capa Oculta
    # Matriz de pesos W1 que conecta 784 pixeles con 512 neuronas ocultas
    # Se resta 0.5 para centrar los valores alrededor de 0
    W1 = np.random.randn(TAMAÑO_ENTRADA, TAMAÑO_CAPA_OCULTA) * np.sqrt(2. / TAMAÑO_ENTRADA)
    b1 = np.zeros((1, TAMAÑO_CAPA_OCULTA))  # Vector de sesgos para capa oculta
    
    # CAPA 2: Capa Oculta a Salida
    # Matriz de pesos W2 que conecta 512 neuronas ocultas con 10 digitos de salida
    W2 = np.random.randn(TAMAÑO_CAPA_OCULTA, TAMAÑO_SALIDA) * np.sqrt(1. / TAMAÑO_CAPA_OCULTA)
    b2 = np.zeros((1, TAMAÑO_SALIDA))  # Vector de sesgos para capa de salida
    
    return W1, b1, W2, b2


W1, b1, W2, b2 = inicializar_parametros()

#endregion

#region Funciones Auxiliares

# Función que recibe una matriz y compara cada número, si es negativo lo vuelve 0, si no lo deja igual
def relu(Z):
    return np.maximum(0, Z)

# Función que convierte números en probabilidades
def softmax(Z):
    # Restar a todos el número máximo para estabilidad numérica (No obtener números muy grandes)
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True)) 
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

def one_hot(y, num_classes):
    # Crear una matriz de ceros con forma (cantidad_ejemplos, num_classes)
    one_hot_y = np.zeros((y.size, num_classes))
    # Pone un 1 en la posición del valor de cada número en el array y
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y

# Función que calcula la pérdida (loss) de entropía cruzada
def calcular_loss(A2, Y):
    """Calcula el promedio de la Pérdida de Entropía Cruzada (Cross-Entropy Loss)."""
    m = Y.size
    Y_one_hot = one_hot(Y, TAMAÑO_SALIDA)
    
    # Asegurar la estabilidad numérica: prevenir log(0)
    A2_clipped = np.clip(A2, 1e-12, 1.0 - 1e-12) 
    
    # Fórmula de Cross-Entropy Loss
    cost = - (1/m) * np.sum(Y_one_hot * np.log(A2_clipped))
    return cost

#endregion

#region Entrenamiento

def propagacion_adelante(X, W1, b1, W2, b2):
    # Capa 1
    # Multiplicación entradas x pesos + sesgo
    Z1 = np.dot(X, W1) + b1
    # Aplicación de la función de activación ReLU
    A1 = relu(Z1)
    
    # Capa 2
    # Multiplicación capa oculta x pesos + sesgo
    Z2 = np.dot(A1, W2) + b2
    # Aplicación de la función de activación Softmax
    A2 = softmax(Z2) # Contiene las predicciones finales (probabilidades)
    
    # Guardar Z1 y A1 en un diccionario tipo caché
    # Es util para la propagación hacia atrás
    cache = {
        'Z1': Z1,
        'A1': A1
    }
    return A2, cache


def backward_pass(X, Y, A2, cache, W2):
    # Recuperr lo que hay en caché
    A1 = cache["A1"]
    Z1 = cache["Z1"]
    m = X.shape[0]  # Obtener el tamaño del batch o lote

    # Convertir etiquetas reales a formato One-Hot
    # Ejemplo: Si Y es 5 -> [0,0,0,0,0,1,0,0,0,0]
    Y_one_hot = one_hot(Y, 10)

    # CAPA SALIDA (Cálculo del error final)
    # Diferencia entre lo que predijo y lo real
    dZ2 = A2 - Y_one_hot
    
    # Gradiente de los pesos de salida
    # Multiplicar la activación anterior (transpuesta) por el error
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    
    # Gradiente del sesgo (promedio de los errores)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    # CAPA OCULTA (Llevar error hacia atrás)
    # Error propagado a la capa 1
    dA1 = np.dot(dZ2, W2.T)
    
    # Derivada de ReLU. 
    # Si Z1 > 0, la derivada es 1 (pasa el error). Si Z1 <= 0, es 0 (bloquea el error).
    dZ1 = dA1 * (Z1 > 0)
    
    # Gradientes de la capa 1
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2


def actualizar_parametros(W1, b1, W2, b2, dW1, db1, dW2, db2, tasa_aprendizaje):
    # Actualizar pesos y sesgos restando el gradiente multiplicado por la tasa de aprendizaje
    W1 -= tasa_aprendizaje * dW1
    b1 -= tasa_aprendizaje * db1
    W2 -= tasa_aprendizaje * dW2
    b2 -= tasa_aprendizaje * db2
    return W1, b1, W2, b2

# endregion

#region Evaluación
print ("Iniciando entrenamiento de la red neuronal...")
inicio_tiempo = time.time()

# Bucle según número de épocas (Veces que se muestra todo el dataset)
for epoca in range(EPOCAS):
    # Mezclar aleatoriamente los datos de entrenamiento
    permutacion = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[permutacion]
    y_shuffled = y_train[permutacion]
    
    # Bucle según tamaño del batch
    for i in range(0, X_train.shape[0], TAMAÑO_BATCH):
        # Se corta un batch de datos
        X_batch = X_shuffled[i:i+TAMAÑO_BATCH]
        y_batch = y_shuffled[i:i+TAMAÑO_BATCH]
        
        # Fase de propagación hacia adelante
        A2, cache = propagacion_adelante(X_batch, W1, b1, W2, b2)
        
        # Fase de propagación hacia atrás
        dW1, db1, dW2, db2 = backward_pass(X_batch, y_batch, A2, cache, W2)
        
        # Actualizar parámetros
        W1, b1, W2, b2 = actualizar_parametros(W1, b1, W2, b2, dW1, db1, dW2, db2, TASA_APRENDIZAJE)
    
    # --- Cálculo y Reporte de Métricas al final de la Época ---
    
    # Volvemos a calcular A2 usando todo el dataset mezclado para métricas de época
    A2_epoca, _ = propagacion_adelante(X_shuffled, W1, b1, W2, b2) 
    
    # Precisión
    precision = np.mean(np.argmax(A2_epoca, axis=1) == y_shuffled)
    
    # Pérdida (Loss)
    loss_valor = calcular_loss(A2_epoca, y_shuffled)
    
    print(f"Época {epoca+1}/{EPOCAS} completada")
    print(f" -> Precisión: {precision:.4f}")
    print(f" -> Loss (Pérdida): {loss_valor:.4f}")
    
fin_tiempo = time.time()
print(f"Entrenamiento finalizado en {fin_tiempo - inicio_tiempo:.2f} segundos.")