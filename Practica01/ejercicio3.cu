/*
Compilación y ejecución:
Para compilar y ejecutar este programa, utiliza el siguiente comando:
nvcc -o ejer3 ejercicio3.cu
./ejer3
*/

// Programa para multiplicar dos matrices en CUDA
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#define TAMANO_BLOQUE 16 // Tamaño del bloque para optimización

// Kernel para multiplicar matrices
__global__ void multiplicarMatrices(const float *ptr_m1, const float *ptr_m2,
                                    float *ptr_res, int n) {
  __shared__ float subMatriz1[TAMANO_BLOQUE][TAMANO_BLOQUE];
  __shared__ float subMatriz2[TAMANO_BLOQUE][TAMANO_BLOQUE];

  int fila = blockIdx.y * blockDim.y + threadIdx.y;    // Fila global
  int columna = blockIdx.x * blockDim.x + threadIdx.x; // Columna global

  float suma = 0.0f;
  for (int m = 0; m < (n + TAMANO_BLOQUE - 1) / TAMANO_BLOQUE; m++) {
    // Cargar submatrices en memoria compartida
    if (fila < n && m * TAMANO_BLOQUE + threadIdx.x < n)
      subMatriz1[threadIdx.y][threadIdx.x] =
          ptr_m1[fila * n + m * TAMANO_BLOQUE + threadIdx.x];
    else
      subMatriz1[threadIdx.y][threadIdx.x] = 0.0f;

    if (columna < n && m * TAMANO_BLOQUE + threadIdx.y < n)
      subMatriz2[threadIdx.y][threadIdx.x] =
          ptr_m2[(m * TAMANO_BLOQUE + threadIdx.y) * n + columna];
    else
      subMatriz2[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads(); // Sincronizar hilos

    // Calcular producto
    for (int k = 0; k < TAMANO_BLOQUE; k++) {
      suma += subMatriz1[threadIdx.y][k] * subMatriz2[k][threadIdx.x];
    }
    __syncthreads(); // Sincronizar hilos
  }

  // Escribir resultado
  if (fila < n && columna < n)
    ptr_res[fila * n + columna] = suma;
}

int main() {
  const int N = 1024; // Tamaño de la matriz (N x N)

  // Inicializar generador de números aleatorios con semilla fija
  std::mt19937 gen(322); // Semilla fija
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  // Crear matrices en el host
  std::vector<float> m1(N * N), m2(N * N), m_res(N * N);
  for (int i = 0; i < N * N; ++i) {
    m1[i] = dist(gen);
    m2[i] = dist(gen);
  }

  // Punteros para memoria en el dispositivo (GPU)
  float *ptr_m1, *ptr_m2, *ptr_res;

  // Asignar memoria en la GPU
  cudaMalloc(&ptr_m1, N * N * sizeof(float));
  cudaMalloc(&ptr_m2, N * N * sizeof(float));
  cudaMalloc(&ptr_res, N * N * sizeof(float));

  // Copiar datos desde el host al dispositivo
  cudaMemcpy(ptr_m1, m1.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(ptr_m2, m2.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

  // Configurar grid y bloque
  dim3 tamanoBloque(TAMANO_BLOQUE, TAMANO_BLOQUE);
  dim3 tamanoGrid((N + TAMANO_BLOQUE - 1) / TAMANO_BLOQUE,
                  (N + TAMANO_BLOQUE - 1) / TAMANO_BLOQUE);

  // Medir tiempo de ejecución del kernel
  auto inicio = std::chrono::high_resolution_clock::now();
  multiplicarMatrices<<<tamanoGrid, tamanoBloque>>>(ptr_m1, ptr_m2, ptr_res, N);
  cudaDeviceSynchronize(); // Esperar a que el kernel termine
  auto fin = std::chrono::high_resolution_clock::now();
  auto duracion =
      std::chrono::duration_cast<std::chrono::microseconds>(fin - inicio);

  // Copiar resultado de la GPU al host
  cudaMemcpy(m_res.data(), ptr_res, N * N * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Imprimir tiempo de ejecución
  std::cout << "Tiempo de ejecución en GPU: " << duracion.count()
            << " microsegundos\n";

  // Mostrar algunos resultados (primera fila)
  std::cout << "Primeros 3 resultados (fila 0):\n";
  for (int i = 0; i < 3 && i < N; ++i) {
    std::cout << "m_res[0][" << i << "] = " << m_res[i] << "\n";
  }

  std::cout << "Últimos 3 resultados (fila 0):\n";
  for (int i = std::max(0, N - 3); i < N; ++i) {
    std::cout << "m_res[0][" << i << "] = " << m_res[i] << "\n";
  }

  // Liberar memoria del dispositivo
  cudaFree(ptr_m1);
  cudaFree(ptr_m2);
  cudaFree(ptr_res);

  return 0;
}
