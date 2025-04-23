/*
Compilación y ejecución:
Para compilar y ejecutar este programa, utiliza el siguiente comando:
nvcc -o ejer1 ejercicio1.cu
./ejer1
*/

// Programa para obtener información específica del dispositivo CUDA
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount); // Obtener número de dispositivos CUDA

  if (deviceCount == 0) {
    printf("No se encontraron dispositivos CUDA.\n");
    return 1;
  }

  // Obtener propiedades del primer dispositivo
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  // Mostrar información en el formato solicitado
  printf("Salida\n");
  printf("Nombre del dispositivo: %s\n", prop.name);
  printf("Memoria global total: %lu\n", prop.totalGlobalMem);
  printf("Número de SMs: %d\n", prop.multiProcessorCount);
  printf("Memoria compartida por SM: %lu\n", prop.sharedMemPerMultiprocessor);
  printf("Registros por SM: %d\n", prop.regsPerMultiprocessor);

  return 0;
}
