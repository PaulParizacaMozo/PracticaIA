/*
Compilación y ejecución:
Para compilar y ejecutar este programa, utiliza el siguiente comando:
nvcc -o ejer2 ejercicio2.cu
./ejer2
*/

// Programa para sumar dos vectores en CUDA
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

// Kernel para sumar vectores
__global__ void sumarVectores(const float *ptr_v1, const float *ptr_v2,
                              float *ptr_res, int n) {
  int indice = blockIdx.x * blockDim.x + threadIdx.x;
  if (indice < n) {
    ptr_res[indice] = ptr_v1[indice] + ptr_v2[indice];
  }
}

int main() {
  const int N = 1000000; // Número de elementos

  // Inicializar generador de números aleatorios con semilla fija
  std::mt19937 gen(322); // Semilla fija
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  // Crear vectores en el host
  std::vector<float> v1(N), v2(N), v_res(N);
  for (int i = 0; i < N; ++i) {
    v1[i] = dist(gen);
    v2[i] = dist(gen);
  }

  // Punteros para memoria en el dispositivo (GPU)
  float *ptr_v1, *ptr_v2, *ptr_res;

  // Asignar memoria en la GPU
  cudaMalloc(&ptr_v1, N * sizeof(float));
  cudaMalloc(&ptr_v2, N * sizeof(float));
  cudaMalloc(&ptr_res, N * sizeof(float));

  // Copiar datos desde el host al dispositivo
  cudaMemcpy(ptr_v1, v1.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(ptr_v2, v2.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  // Configurar grid y bloque
  int tamanoBloque = 256;
  int tamanoGrid = (N + tamanoBloque - 1) / tamanoBloque;

  // Medir tiempo de ejecución del kernel
  auto inicio = std::chrono::high_resolution_clock::now();
  sumarVectores<<<tamanoGrid, tamanoBloque>>>(ptr_v1, ptr_v2, ptr_res, N);
  cudaDeviceSynchronize(); // Esperar a que el kernel termine
  auto fin = std::chrono::high_resolution_clock::now();
  auto duracion =
      std::chrono::duration_cast<std::chrono::microseconds>(fin - inicio);

  // Copiar resultado de la GPU al host
  cudaMemcpy(v_res.data(), ptr_res, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Imprimir tiempo de ejecución
  std::cout << "Tiempo de ejecución en GPU: " << duracion.count()
            << " microsegundos\n";

  // Mostrar algunos resultados
  std::cout << "Primeros 3 resultados:\n";
  for (int i = 0; i < 3 && i < N; ++i) {
    std::cout << v1[i] << " + " << v2[i] << " = " << v_res[i] << "\n";
  }

  std::cout << "Últimos 3 resultados:\n";
  for (int i = std::max(0, N - 3); i < N; ++i) {
    std::cout << v1[i] << " + " << v2[i] << " = " << v_res[i] << "\n";
  }

  // Liberar memoria del dispositivo
  cudaFree(ptr_v1);
  cudaFree(ptr_v2);
  cudaFree(ptr_res);

  return 0;
}
