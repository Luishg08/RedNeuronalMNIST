#include <stdio.h>  // Leer y escribir en consola
#include <stdlib.h> // Funciones generales de utilidad
#include <math.h>   // Funciones matemáticas
#include <time.h>   // Medición de tiempo
#include <string.h> // Manipulación de cadenas

#pragma region Clases y Estructuras

typedef struct
{
    int filas;
    int columnas;
    float *datos; // Puntero a los datos de la matriz en una sola dimensión
} Matriz;

// Crear una matriz con dimensiones dadas y reservar memoria
Matriz *crear_matriz(int filas, int columnas)
{
    Matriz *m = (Matriz *)malloc(sizeof(Matriz));
    m->filas = filas;
    m->columnas = columnas;
    m->datos = (float *)malloc(filas * columnas * sizeof(float));
    return m;
}

#pragma endregion

#pragma region Manipular Matrices

// Liberar la memoria ocupada por una matriz
void liberar_matriz(Matriz *m)
{
    if (m)
    {
        if (m->datos)
            free(m->datos);
        free(m);
    }
}

// Inicializar una matriz con valores aleatorios entre -0.5 y 0.5
void inicializar_matriz_aleatoria(Matriz *m)
{
    for (int i = 0; i < m->filas * m->columnas; i++)
    {
        m->datos[i] = ((float)rand() / RAND_MAX) - 0.5f; // Valores entre -0.5 y 0.5
    }
}

// Llenar una matriz con limpiar_matriz
void limpiar_matriz(Matriz *m)
{
    memset(m->datos, 0, m->filas * m->columnas * sizeof(float));
}

// Multiplicar dos matrices A y B, almacenar el resultado en C
void multiplicar_matrices(Matriz *A, Matriz *B, Matriz *C)
{
    if (A->columnas != B->filas)
    {
        printf("Error: dimensiones no compatibles para multiplicación de matrices.\n");
        exit(1);
    }
    for (int i = 0; i < A->filas; i++)
    {
        for (int j = 0; j < B->columnas; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < A->columnas; k++)
            {
                sum += A->datos[i * A->columnas + k] * B->datos[k * B->columnas + j];
            }
            C->datos[i * C->columnas + j] = sum;
        }
    }
}

// Calcular la transpuesta de una matriz A y almacenarla en B
// Asume que B ya ha sido inicializada con las dimensiones correctas (Dimensiones invertidas de A)
// Se utiliza para la propagación hacia atrás en redes neuronales: dW = A^T * dZ
void transpuesta(Matriz *A, Matriz *B)
{
    for (int i = 0; i < A->filas; i++)
    {
        for (int j = 0; j < A->columnas; j++)
        {
            // La fila i, col j de A pasa a ser la fila j, col i de B
            B->datos[j * B->columnas + i] = A->datos[i * A->columnas + j];
        }
    }
}

void sumar_sesgo(Matriz *m, Matriz *b)
{
    for (int i = 0; i < m->filas; i++)
    {
        for (int j = 0; j < m->columnas; j++)
        {
            m->datos[i * m->columnas + j] += b->datos[j];
        }
    }
}
// Multiplicar una matriz por un escalar
void multiplicar_escalar(Matriz *m, float escalar)
{
    for (int i = 0; i < m->filas * m->columnas; i++)
    {
        m->datos[i] *= escalar;
    }
}

// Restar matrices A = A - B
void restar_matrices(Matriz *A, Matriz *B)
{
    for (int i = 0; i < A->filas * A->columnas; i++)
    {
        A->datos[i] -= B->datos[i];
    }
}

#pragma endregion

#pragma region Funciones Auxiliares

// Función de activación ReLU: Si x < 0, devuelve 0; si x >= 0, devuelve x
void relu(Matriz *m)
{
    for (int i = 0; i < m->filas * m->columnas; i++)
    {
        if (m->datos[i] < 0)
            m->datos[i] = 0;
    }
}

// Softmax: Convierte números en probabilidades
void softmax(Matriz *m)
{
    for (int i = 0; i < m->filas; i++)
    {
        // Buscar máximo de la fila (estabilidad numérica)
        float max_val = -1e9;
        for (int j = 0; j < m->columnas; j++)
        {
            if (m->datos[i * m->columnas + j] > max_val)
                max_val = m->datos[i * m->columnas + j];
        }
        // Exponencial y suma
        float sum = 0.0f;
        for (int j = 0; j < m->columnas; j++)
        {
            m->datos[i * m->columnas + j] = expf(m->datos[i * m->columnas + j] - max_val);
            sum += m->datos[i * m->columnas + j];
        }
        // Normalizar para obtener probabilidades
        for (int j = 0; j < m->columnas; j++)
        {
            m->datos[i * m->columnas + j] /= sum;
        }
    }
}

// Encuentra el índice del valor más alto (Argmax)
// Ej: Si la salida es [0.1, 0.8, 0.1], devuelve 1.
int argmax(Matriz *m, int row)
{
    float max_val = -1e9;
    int max_index = 0;
    for (int j = 0; j < m->columnas; j++)
    {
        if (m->datos[row * m->columnas + j] > max_val)
        {
            max_val = m->datos[row * m->columnas + j];
            max_index = j;
        }
    }
    return max_index;
}

#pragma endregion

#pragma region Cargar Datos

// Convertir Bytes High Endian a Enteros
int convertir_bytes_a_enteros(FILE *fp)
{
    unsigned char buf[4];
    if (fread(buf, sizeof(unsigned char), 4, fp) != 4)
        return 0;
    return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
}

Matriz *cargar_imagenes_dataset(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Error abriendo %s\n", filename);
        exit(1);
    }

    convertir_bytes_a_enteros(fp);
    int num_imgs = convertir_bytes_a_enteros(fp);
    int rows = convertir_bytes_a_enteros(fp);
    int cols = convertir_bytes_a_enteros(fp);

    Matriz *m = crear_matriz(num_imgs, rows * cols);
    unsigned char temp;
    for (int i = 0; i < m->filas * m->columnas; i++)
    {
        // Leer un byte por píxel
        fread(&temp, sizeof(unsigned char), 1, fp);
        // División por 255 para normalizar valores entre 0 y 1
        m->datos[i] = (float)temp / 255.0f;
    }
    fclose(fp);
    return m;
}

Matriz *cargar_etiquetas_dataset(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Error abriendo %s\n", filename);
        exit(1);
    }

    convertir_bytes_a_enteros(fp);
    int num_items = convertir_bytes_a_enteros(fp);

    Matriz *m = crear_matriz(num_items, 1);
    unsigned char temp;
    for (int i = 0; i < num_items; i++)
    {
        fread(&temp, sizeof(unsigned char), 1, fp);
        m->datos[i] = (float)temp;
    }
    fclose(fp);
    return m;
}

#pragma endregion

#pragma region Entrenamiento

int main()
{
    srand(time(NULL)); // Semilla para números aleatorios

    const int TAMAÑO_ENTRADA = 784; // 28x28 píxeles
    const int TAMAÑO_CAPA_OCULTA = 512;
    const int TAMAÑO_SALIDA = 10; // Dígitos 0-9
    const float TASA_APRENDIZAJE = 0.01f;
    const int EPOCAS = 10;
    const int TAMAÑO_BATCH = 64;

    printf("Cargando datos...\n");
    Matriz *X_train = cargar_imagenes_dataset("./Resources/train-images.idx3-ubyte");
    Matriz *Y_train = cargar_etiquetas_dataset("./Resources/train-labels.idx1-ubyte");
    printf("Entrenamiento: %d imagenes cargadas.\n", X_train->filas);

    // Inicializar pesos y sesgos
    Matriz *W1 = crear_matriz(TAMAÑO_ENTRADA, TAMAÑO_CAPA_OCULTA);
    inicializar_matriz_aleatoria(W1);
    Matriz *b1 = crear_matriz(1, TAMAÑO_CAPA_OCULTA);
    limpiar_matriz(b1);
    Matriz *W2 = crear_matriz(TAMAÑO_CAPA_OCULTA, TAMAÑO_SALIDA);
    inicializar_matriz_aleatoria(W2);
    Matriz *b2 = crear_matriz(1, TAMAÑO_SALIDA);
    limpiar_matriz(b2);

    // Reservar memoria
    Matriz *X_batch = crear_matriz(TAMAÑO_BATCH, TAMAÑO_ENTRADA);
    Matriz *Y_batch = crear_matriz(TAMAÑO_BATCH, 1);

    // Temporales de propagación hacia adelante
    Matriz *Z1 = crear_matriz(TAMAÑO_BATCH, TAMAÑO_CAPA_OCULTA);
    Matriz *A1 = crear_matriz(TAMAÑO_BATCH, TAMAÑO_CAPA_OCULTA);
    Matriz *Z2 = crear_matriz(TAMAÑO_BATCH, TAMAÑO_SALIDA);
    Matriz *A2 = crear_matriz(TAMAÑO_BATCH, TAMAÑO_SALIDA);

    // Temporales de propagación hacia atrás
    Matriz *dZ2 = crear_matriz(TAMAÑO_BATCH, TAMAÑO_SALIDA);
    Matriz *dW2 = crear_matriz(TAMAÑO_CAPA_OCULTA, TAMAÑO_SALIDA);
    Matriz *db2 = crear_matriz(1, TAMAÑO_SALIDA);
    Matriz *A1_T = crear_matriz(TAMAÑO_CAPA_OCULTA, TAMAÑO_BATCH);

    Matriz *dZ1 = crear_matriz(TAMAÑO_BATCH, TAMAÑO_CAPA_OCULTA);
    Matriz *dW1 = crear_matriz(TAMAÑO_ENTRADA, TAMAÑO_CAPA_OCULTA);
    Matriz *db1 = crear_matriz(1, TAMAÑO_CAPA_OCULTA);
    Matriz *W2_T = crear_matriz(TAMAÑO_SALIDA, TAMAÑO_CAPA_OCULTA);
    Matriz *X_batch_T = crear_matriz(TAMAÑO_ENTRADA, TAMAÑO_BATCH);

    // Indices para mezclar datos
    int *indices = (int *)malloc(X_train->filas * sizeof(int));
    for (int k = 0; k < X_train->filas; k++)
        indices[k] = k;

    // Medición de tiempo
    clock_t inicio_tiempo = clock();
    // Bucle de entrenamiento
    for (int epoca = 0; epoca < EPOCAS; epoca++)
    {
        // Mezclar datos al inicio de cada época
        for (int k = X_train->filas - 1; k > 0; k--)
        {
            int j = rand() % (k + 1);
            int temp = indices[k];
            indices[k] = indices[j];
            indices[j] = temp;
        }

        for (int i = 0; i < X_train->filas; i += TAMAÑO_BATCH)
        {
            int batch_actual = (i + TAMAÑO_BATCH > X_train->filas) ? X_train->filas - i : TAMAÑO_BATCH;
            // Preparar Batch
            for (int b = 0; b < batch_actual; b++)
            {
                int idx = indices[i + b];
                memcpy(&X_batch->datos[b * TAMAÑO_ENTRADA], &X_train->datos[idx * TAMAÑO_ENTRADA], TAMAÑO_ENTRADA * sizeof(float));
                Y_batch->datos[b] = Y_train->datos[idx];
            }
            // Ajustar tamaño lógico de matrices batch (por si el último es menor)
            X_batch->filas = batch_actual;
            Z1->filas = batch_actual;
            A1->filas = batch_actual;
            Z2->filas = batch_actual;
            A2->filas = batch_actual;

            // Propagación hacia adelante
            multiplicar_matrices(X_batch, W1, Z1);
            sumar_sesgo(Z1, b1);
            memcpy(A1->datos, Z1->datos, batch_actual * TAMAÑO_CAPA_OCULTA * sizeof(float));
            relu(A1);

            multiplicar_matrices(A1, W2, Z2);
            sumar_sesgo(Z2, b2);
            memcpy(A2->datos, Z2->datos, batch_actual * TAMAÑO_SALIDA * sizeof(float));
            softmax(A2);

            // Propagación hacia atrás
            // dZ2 = A2 - Y (one-hot)
            memcpy(dZ2->datos, A2->datos, batch_actual * TAMAÑO_SALIDA * sizeof(float));
            for (int b = 0; b < batch_actual; b++)
            {
                dZ2->datos[b * TAMAÑO_SALIDA + (int)Y_batch->datos[b]] -= 1.0f;
            }

            // Grads Capa 2
            A1->filas = batch_actual; // Asegurar dim correcta
            transpuesta(A1, A1_T);
            multiplicar_matrices(A1_T, dZ2, dW2);
            multiplicar_escalar(dW2, 1.0f / batch_actual);

            limpiar_matriz(db2);
            for (int r = 0; r < batch_actual; r++)
            {
                for (int c = 0; c < TAMAÑO_SALIDA; c++)
                    db2->datos[c] += dZ2->datos[r * TAMAÑO_SALIDA + c];
            }
            multiplicar_escalar(db2, 1.0f / batch_actual);

            // Error Capa 1
            transpuesta(W2, W2_T);
            multiplicar_matrices(dZ2, W2_T, dZ1);
            for (int k = 0; k < batch_actual * TAMAÑO_CAPA_OCULTA; k++)
            {
                if (Z1->datos[k] <= 0)
                    dZ1->datos[k] = 0.0f; // Derivada ReLU
            }

            // Grads Capa 1
            X_batch->filas = batch_actual;
            transpuesta(X_batch, X_batch_T);
            multiplicar_matrices(X_batch_T, dZ1, dW1);
            multiplicar_escalar(dW1, 1.0f / batch_actual);

            limpiar_matriz(db1);
            for (int r = 0; r < batch_actual; r++)
            {
                for (int c = 0; c < TAMAÑO_CAPA_OCULTA; c++)
                    db1->datos[c] += dZ1->datos[r * TAMAÑO_CAPA_OCULTA + c];
            }
            multiplicar_escalar(db1, 1.0f / batch_actual);

            // Actualizar parámetros
            multiplicar_escalar(dW1, TASA_APRENDIZAJE);
            restar_matrices(W1, dW1);
            multiplicar_escalar(db1, TASA_APRENDIZAJE);
            restar_matrices(b1, db1);
            multiplicar_escalar(dW2, TASA_APRENDIZAJE);
            restar_matrices(W2, dW2);
            multiplicar_escalar(db2, TASA_APRENDIZAJE);
            restar_matrices(b2, db2);
        }
        printf("Epoca %d completada.\n", epoca + 1);
    }
#pragma endregion
#pragma region Evaluación
    double total_time = (double)(clock() - inicio_tiempo) / CLOCKS_PER_SEC;
    printf("Entrenamiento C finalizado en %.2f segundos.\n", total_time);

#pragma endregion

#pragma region Limpiar Memoria
    free(indices);
    liberar_matriz(X_train);
    liberar_matriz(Y_train);
    liberar_matriz(W1);
    liberar_matriz(b1);
    liberar_matriz(W2);
    liberar_matriz(b2);
    liberar_matriz(X_batch);
    liberar_matriz(Y_batch);
    liberar_matriz(Z1);
    liberar_matriz(A1);
    liberar_matriz(Z2);
    liberar_matriz(A2);
    liberar_matriz(dZ2);
    liberar_matriz(dW2);
    liberar_matriz(db2);
    liberar_matriz(A1_T);
    liberar_matriz(dZ1);
    liberar_matriz(dW1);
    liberar_matriz(db1);
    liberar_matriz(W2_T);
    liberar_matriz(X_batch_T);
#pragma endregion
    return 0;
}

#pragma region Ejecución
// Ejecutar con:
// gcc mainB.c -o mlp -lm -O3
// ./mlp
#pragma endregion