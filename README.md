# Red Neuronal MNIST

Implementación de una red neuronal para clasificación de dígitos del dataset MNIST en escenarios: Python secuencial, C secuencial, Python con Multiprocessing, C con OpenMP y CUDA.

## Autores

**Luis Miguel Henao** y **Mariana López**  
Universidad de Caldas  
Programación Concurrente Y Distribuida

## Estructura del Proyecto

```
ProyectoRedNeuronal/
├── Resources/              # Dataset MNIST
├── Escenario1/            # Python y C secuencial
├── Escenario2/            # C con OpenMP y Python con Multiprocessing
└── Escenario3/            # CUDA
```

## Requisitos

### Python (Escenario 1)
- Python 3.x
- NumPy

### C (Escenarios 1 y 2)
- GCC con soporte para OpenMP
- Librerías: `math.h`, `omp.h`

### CUDA (Escenario 3)
- NVIDIA CUDA Toolkit
- GPU compatible con CUDA

## Compilación y Ejecución

**Importante:** Todos los comandos deben ejecutarse desde la raíz del proyecto.

### Escenario 1 - Python (Secuencial)

```bash
python ./Escenario1/mainA.py
```

### Escenario 1 - C (Secuencial)

```bash
gcc ./Escenario1/mainB.c -o ./Escenario1/mlp -lm -O3
./Escenario1/mlp
```

En Windows:
```bash
gcc ./Escenario1/mainB.c -o ./Escenario1/mlp -lm -O3
./Escenario1/mlp.exe
```

### Escenario 2 - Python (Multiprocessing)

```bash
python ./Escenario2/mainA.py
```

### Escenario 2 - C con OpenMP (Paralelo)

```bash
gcc ./Escenario2/mainB.c -o ./Escenario2/mlp -lm -fopenmp -O3
./Escenario2/mlp
```

En Windows:
```bash
gcc ./Escenario2/mainB.c -o ./Escenario2/mlp -lm -fopenmp -O3
./Escenario2/mlp.exe
```

## Ejecución de CUDA

En Google Colab, subir el archivo `mainA.ipynb` y la carpeta `Resources/`, luego ejecutar las celdas del notebook.

- Los archivos del dataset MNIST deben estar en la carpeta `Resources/`
- El flag `-O3` optimiza el código para mejor rendimiento
- El flag `-fopenmp` habilita la paralelización con OpenMP en el Escenario 2
