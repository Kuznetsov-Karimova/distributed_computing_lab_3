#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>

const float G = 6.67e-11;
const float dt = 0.01;
const float final_time = 100.0f;

__host__ void WriteState(FILE *out, int n, float time, float *positions) {
    fprintf(out, "%f", time);
    for (int index_point = 0; index_point < n; ++index_point) {
        fprintf(out, ",%f,%f", positions[2 * index_point],  positions[2 * index_point + 1]);
    }
    fprintf(out, "\n");
}

__global__ void CalcForces(int n, float *m, float *positions, float *totalF) {
    int first_point = blockDim.x * blockIdx.x + threadIdx.x;
    int second_point = blockDim.y * blockIdx.y + threadIdx.y;
    if (first_point >= second_point || first_point >= n) {
        return;
    }
    float dist_x = positions[first_point * 2] - positions[second_point * 2];
    float dist_y = positions[first_point * 2 + 1] - positions[second_point * 2 + 1];
    float norm = powf(sqrtf(dist_x * dist_x + dist_y * dist_y), 3.0f) + 1e-12;
    float f_coef = G * m[first_point] * m[second_point] / norm;
    atomicAdd(&totalF[2 * first_point], dist_x * f_coef);
    atomicAdd(&totalF[2 * first_point + 1], dist_y * f_coef);
    atomicAdd(&totalF[2 * second_point], - dist_x * f_coef);
    atomicAdd(&totalF[2 * second_point + 1], - dist_y * f_coef);
}

__global__ void CalcState(int n, float* m, float *positions, float *V, float *F) {
    int point_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (point_index >= n) {
        return;
    }
    int point_coord = threadIdx.y;
    int i = 2 * point_index + point_coord;
    positions[i] += V[i] * dt;
    V[i] += F[i] / m[point_index] * dt;
    F[i] = 0.0f;
}

int main() {

    int n;
    unsigned int thread_count;

    printf("Count of points: ");
    scanf("%d", &n);
    printf("Count of thread: ");
    scanf("%d", &thread_count);

    unsigned int block_count = (n + thread_count - 1) / thread_count;

    float *m = (float*) malloc(n * sizeof(float));
    float *positions = (float*) malloc(2 * n * sizeof(float));
    float *V = (float*) malloc(2 * n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        m[i] = (float) (rand() % 1000000) + 1;
        positions[2 * i] = (float) (rand() % 100 - 50);
        positions[2 * i + 1] = (float) (rand() % 100 - 50);
        V[2 * i] = (float) (rand() % 10 - 5);
        V[2 * i + 1] = (float) (rand() % 10 - 5);
    }

    float* d_m, *d_positions, *d_V, *d_F;
    cudaMalloc(&d_m, n * sizeof(float));
    cudaMalloc(&d_positions, 2 * n * sizeof(float));
    cudaMalloc(&d_V, 2 * n * sizeof(float));
    cudaMalloc(&d_F, 2 * n * sizeof(float));

    cudaMemcpy(d_m, m, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, positions, 2 * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, 2 * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_F, 0.0f, 2 * n * sizeof(float));

    FILE *out = fopen("output.csv", "w");
    if (!out) {
        printf("Error with opening output file");
        return -1;
    }

    fprintf(out, "t");
    for (int index_point = 0; index_point < n; ++index_point) {
        fprintf(out, ",x_%d,y_%d", index_point,  index_point);
    }
    fprintf(out, "\n");

    WriteState(out, n, 0.0f, positions);

    dim3 point_size = {thread_count, 2};
    dim3 forces_block_size = {thread_count, thread_count};

    for (float time = 0; time < final_time; time += dt) {
        CalcForces<<<block_count, forces_block_size>>>(n, d_m, d_positions, d_F);
        CalcState<<<block_count, point_size>>>(n, d_m, d_positions, d_V, d_F);
        cudaMemcpy(positions, d_positions, 2 * n * sizeof(float), cudaMemcpyDeviceToHost);
        WriteState(out, n, time + dt, positions);
    }

    cudaFree(d_m);
    cudaFree(d_positions);
    cudaFree(d_V);
    cudaFree(d_F);

    return 0;
}