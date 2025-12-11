import numpy as np
import os
import time
import multiprocessing as mp
import gzip

#region Constantes
TAMAÑO_ENTRADA = 784
TAMAÑO_CAPA_OCULTA = 512
TAMAÑO_SALIDA = 10
#endregion

#region Configuración Entrenamiento
TASA_APRENDIZAJE = 0.01
EPOCAS = 10
TAMAÑO_BATCH = 512  # Según la ejecución reportada
N_WORKERS = mp.cpu_count()
#endregion

#region Carga datos
def cargar_imagenes_dataset(nombre_archivo):
    """Carga imágenes del dataset MNIST comprimido con gzip."""
    with gzip.open(nombre_archivo, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, TAMAÑO_ENTRADA).astype(np.float32) / 255.0

def cargar_etiquetas_dataset(nombre_archivo):
    """Carga etiquetas del dataset MNIST comprimido con gzip."""
    with gzip.open(nombre_archivo, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data
#endregion

#region Inicialización y Auxiliares
def inicializar_parametros():
    """Inicializa pesos y sesgos de la red neuronal."""
    np.random.seed(42)
    W1 = np.random.randn(TAMAÑO_ENTRADA, TAMAÑO_CAPA_OCULTA) * np.sqrt(2. / TAMAÑO_ENTRADA)
    b1 = np.zeros((1, TAMAÑO_CAPA_OCULTA))
    W2 = np.random.randn(TAMAÑO_CAPA_OCULTA, TAMAÑO_SALIDA) * np.sqrt(1. / TAMAÑO_CAPA_OCULTA)
    b2 = np.zeros((1, TAMAÑO_SALIDA))
    return W1, b1, W2, b2

def relu(Z): return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

def one_hot(y, num_classes):
    one_hot_y = np.zeros((y.size, num_classes))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y

def calcular_loss(A2, Y):
    """Calcula el promedio de la Pérdida de Entropía Cruzada (Cross-Entropy Loss)."""
    m = Y.size
    Y_one_hot = one_hot(Y, TAMAÑO_SALIDA)
    A2_clipped = np.clip(A2, 1e-12, 1.0 - 1e-12)
    cost = - (1/m) * np.sum(Y_one_hot * np.log(A2_clipped))
    return cost
#endregion

#region Lógica Neuronal (Funciones base)

def propagacion_adelante(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    cache = {'Z1': Z1, 'A1': A1}
    return A2, cache

def backward_pass(X, Y, A2, cache, W2):
    """Calcula la SUMA de los gradientes sobre el sub-lote (sin promediar)."""
    A1 = cache["A1"]
    Z1 = cache["Z1"]
    Y_one_hot = one_hot(Y, 10)
    
    # 1. Gradiente de la salida
    dZ2 = A2 - Y_one_hot
    
    # 2. Gradientes de W2 y b2 (Suma)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # 3. Gradiente de la capa oculta
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    
    # 4. Gradientes de W1 y b1 (Suma)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2 # Devuelve la SUMA de gradientes

def actualizar_parametros(W1, b1, W2, b2, dW1, db1, dW2, db2, tasa_aprendizaje):
    """Aplica el paso de descenso de gradiente."""
    W1 -= tasa_aprendizaje * dW1
    b1 -= tasa_aprendizaje * db1
    W2 -= tasa_aprendizaje * dW2
    b2 -= tasa_aprendizaje * db2
    return W1, b1, W2, b2
#endregion

#region Función de Entrenamiento del Worker
def calcular_gradiente_lote(X_sub_batch, y_sub_batch, W1, b1, W2, b2):
    """
    Función que ejecuta un worker. Calcula el gradiente (suma) 
    sobre su porción de datos.
    """
    A2, cache = propagacion_adelante(X_sub_batch, W1, b1, W2, b2)
    # backward_pass devuelve la SUMA de gradientes
    dW1, db1, dW2, db2 = backward_pass(X_sub_batch, y_sub_batch, A2, cache, W2)
    return dW1, db1, dW2, db2
#endregion

#region Bucle de Entrenamiento Principal

def entrenar_red():
    # --- Carga de Datos y Rutas Robustas ---
    print("Cargando dataset MNIST...")
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_resources = os.path.join(os.path.dirname(ruta_script), 'Resources')
    
    # Rutas corregidas para usar gzip
    X_train = cargar_imagenes_dataset(os.path.join(ruta_resources, 'train-images-idx3-ubyte.gz'))
    y_train = cargar_etiquetas_dataset(os.path.join(ruta_resources, 'train-labels-idx1-ubyte.gz'))
    X_test = cargar_imagenes_dataset(os.path.join(ruta_resources, 't10k-images-idx3-ubyte.gz'))
    y_test = cargar_etiquetas_dataset(os.path.join(ruta_resources, 't10k-labels-idx1-ubyte.gz'))

    print(f"Dataset cargado con {X_train.shape[0]} imágenes de entrenamiento y {X_test.shape[0]} imágenes de prueba.")
    print(f"Usando {N_WORKERS} procesos en paralelo.")
    
    W1, b1, W2, b2 = inicializar_parametros()
    print("Iniciando entrenamiento de la red neuronal...")
    inicio_tiempo = time.time()

    # FIX CRÍTICO: Inicializar el Pool UNA SOLA VEZ fuera del bucle de batches
    with mp.get_context('spawn').Pool(processes=N_WORKERS) as pool: 

        for epoca in range(EPOCAS):
            permutacion = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutacion]
            y_shuffled = y_train[permutacion]

            for i in range(0, X_train.shape[0], TAMAÑO_BATCH):
                X_batch = X_shuffled[i:i+TAMAÑO_BATCH]
                y_batch = y_shuffled[i:i+TAMAÑO_BATCH]
                
                # Capturamos el tamaño real del batch (necesario para el promedio)
                M_BATCH = X_batch.shape[0]

                # 1. División de datos
                X_sub_batches = np.array_split(X_batch, N_WORKERS)
                y_sub_batches = np.array_split(y_batch, N_WORKERS)

                # 2. Creación de argumentos (Paso de pesos por copia)
                args = [(X_sub_batches[j], y_sub_batches[j], W1, b1, W2, b2)
                         for j in range(N_WORKERS)]

                # 3. Ejecución en paralelo y Recolección de Gradientes (SUMAS)
                gradientes_locales = pool.starmap(calcular_gradiente_lote, args)

                # 4. Agregación (Suma) de Gradientes Locales
                dW1_global = np.zeros_like(W1)
                db1_global = np.zeros_like(b1)
                dW2_global = np.zeros_like(W2)
                db2_global = np.zeros_like(b2)

                for dW1_loc, db1_loc, dW2_loc, db2_loc in gradientes_locales:
                    dW1_global += dW1_loc
                    db1_global += db1_loc
                    dW2_global += dW2_loc
                    db2_global += db2_loc
                
                # 5. Normalización Global (Promedio Correcto)
                # Paso CRÍTICO: Dividir la SUMA GLOBAL por el tamaño total del batch (M_BATCH)
                dW1_global /= M_BATCH
                db1_global /= M_BATCH
                dW2_global /= M_BATCH
                db2_global /= M_BATCH

                # 6. Actualizar parámetros (El Maestro)
                W1, b1, W2, b2 = actualizar_parametros(W1, b1, W2, b2,
                                                        dW1_global, db1_global, dW2_global, db2_global,
                                                        TASA_APRENDIZAJE)

            # --- Cálculo y Reporte de Métricas al final de la Época ---
            # Volvemos a calcular A2 usando todo el dataset mezclado para métricas de época
            A2_epoca, _ = propagacion_adelante(X_shuffled, W1, b1, W2, b2)
            
            # Precisión
            precision_train = np.mean(np.argmax(A2_epoca, axis=1) == y_shuffled)
            
            # Pérdida (Loss)
            loss_train = calcular_loss(A2_epoca, y_shuffled)
            
            print(f"Época {epoca+1}/{EPOCAS} completada")
            print(f"  -> Precisión Entrenamiento: {precision_train:.4f}")
            print(f"  -> Loss Entrenamiento: {loss_train:.4f}")

    fin_tiempo = time.time()
    print(f"\nEntrenamiento finalizado en {fin_tiempo - inicio_tiempo:.2f} segundos con {N_WORKERS} workers.")
    
    # --- Evaluación Final en Conjunto de Prueba ---
    print("\n--- Evaluando en el conjunto de prueba ---")
    A2_test, _ = propagacion_adelante(X_test, W1, b1, W2, b2)
    precision_test = np.mean(np.argmax(A2_test, axis=1) == y_test)
    loss_test = calcular_loss(A2_test, y_test)
    
    print(f"Precisión Final (Test): {precision_test:.4f} ({int(precision_test * 100)}%)")
    print(f"Loss Final (Test): {loss_test:.4f}")
    print(f"Imágenes correctas: {int(precision_test * len(y_test))}/{len(y_test)}")

#endregion

# Punto de entrada del programa
if __name__ == '__main__':
    # Es obligatorio incluir esta guarda para que multiprocessing funcione correctamente
    entrenar_red()