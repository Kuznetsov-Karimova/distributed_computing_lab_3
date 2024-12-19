#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>

const float G = 6.67e-11;
const float dt = 0.01;
const float final_time = 100.0f;

void WriteState(FILE *out, int n, float time, float *positions) {
    fprintf(out, "%f", time);
    for (int index_point = 0; index_point < n; ++index_point) {
        fprintf(out, ",%f,%f", positions[2 * index_point],  positions[2 * index_point + 1]);
    }
    fprintf(out, "\n");
}

void CalcForces(int n, float *m, float *positions, float *totalF) {
    
    for (int first_point = 0; first_point < n; ++first_point) 
        for (int second_point = 0; second_point < i; ++second_point) {
            float dist_x = positions[first_point * 2] - positions[second_point * 2];
            float dist_y = positions[first_point * 2 + 1] - positions[second_point * 2 + 1];
            float norm = powf(sqrtf(dist_x * dist_x + dist_y * dist_y), 3.0f) + 1e-12;
            float f_coef = G * m[first_point] * m[second_point] / norm;
            totalF[2 * first_point] += dist_x * f_coef;
            totalF[2 * first_point + 1] += dist_y * f_coef;
            totalF[2 * second_point] -= dist_x * f_coef;
            totalF[2 * second_point + 1] -= dist_y * f_coef;
        }
}

void CalcState(int n, float* m, float *positions, float *V, float *F) {
    for (int i = 0; i < 2 * n; ++i) {
        positions[i] += V[i] * dt;
        V[i] += F[i] / m[i / 2] * dt;
        F[i] = 0.0f;
    }
}

int main() {
    int n;
    unsigned int thread_count;

    FILE *input = fopen("input_10_points.txt", "r");
    if (!input) {
        printf("Error opening input file");
        return -1;
    }

    fscanf(input, "%d", &n);

    printf("Count of thread: ");
    scanf("%d", &thread_count);

    unsigned int block_count = (n + thread_count - 1) / thread_count;

    float *m = (float*) malloc(n * sizeof(float));
    float *positions = (float*) malloc(2 * n * sizeof(float));
    float *V = (float*) malloc(2 * n * sizeof(float));
    float *F = (float*) malloc(2 * n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        fscanf(input, "%f %f %f %f %f", &m[i], &positions[2 * i], &positions[2 * i + 1], &V[2 * i], &V[2 * i + 1]);
    }

    fclose(input);

    memset(F, 0.0f, 2 * n * sizeof(float));

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

    int write_flag = 0;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    dim3 point_size = {thread_count, 2};
    dim3 forces_block_size = {thread_count, thread_count};

    for (float time = 0; time < final_time; time += dt) {
        CalcForces(n, d_m, d_positions, d_F);
        CalcState(n, d_m, d_positions, d_V, d_F);
        if (write_flag) { WriteState(out, n, time + dt, positions); }
    }

    gettimeofday(&end_time, NULL);

    double time_taken = (end_time.tv_sec - start_time.tv_sec) * 1e6;
    time_taken = (time_taken + (end_time.tv_usec - start_time.tv_usec)) * 1e-6;

    printf("Time taken: %f seconds\n", time_taken);

    fclose(out);

    free(m);
    free(positions);
    free(V);

    cudaFree(d_m);
    cudaFree(d_positions);
    cudaFree(d_V);
    cudaFree(d_F);

    return 0;
}
