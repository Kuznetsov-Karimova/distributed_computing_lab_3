#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// #define X0 0.5  // Центр источника
// #define Y0 0.5  // Центр источника
// #define A 1.0   // Амплитуда источника
// #define SIGMA 0.1  // Распределение источника
#define C 0.0  // Граничные условия
#define EPS 1e-6  // Условие сходимости
#define h 0.001

int N;  // Размер сетки
double *memory_block, **mat_U, **mat_F, **mat_U_new;  // Решения, источник и новый массив для обновлений

// Функция для задания источника
double func_f(int i, int j) {
    return 100.;  // простое постоянное значение
}

// Функция для задания начальных значений
double func_u(int i, int j) {
    return 100.;
}

void init() {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Задание значений на границе
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                mat_U[i][j] = C;  // Граничные условия
            } else {
                mat_U[i][j] = func_u(i, j);  // Начальные значения
            }
            mat_F[i][j] = func_f(i, j);  // Источник
        }
    }
}

void allocate_memory() {
    memory_block = (double *)malloc(N * N * 3 * sizeof(double));
    if (!memory_block) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    mat_U = (double **)malloc(N * sizeof(double *));
    mat_F = (double **)malloc(N * sizeof(double *));
    mat_U_new = (double **)malloc(N * sizeof(double *));  

    for (int i = 0; i < N; i++) {
        mat_U[i] = memory_block + i * N;
        mat_F[i] = memory_block + N * N + i * N;
        mat_U_new[i] = memory_block + 2 * N * N + i * N; 
    }
}

void free_memory() {
    free(memory_block);
    free(mat_U);
    free(mat_F);
    free(mat_U_new);
}

int main(int argc, char *argv[]) {
    int numberOfThreads = strtol(argv[1], NULL, 10);
    omp_set_dynamic(0);
    omp_set_num_threads(numberOfThreads);

    printf("Enter N: ");
    scanf("%d", &N);

    allocate_memory();

    double max_diff = 0, diff, max_diff_iter;
    double start_time, end_time;

    init();

    start_time = omp_get_wtime();

    omp_lock_t dmax_lock;
    omp_init_lock(&dmax_lock);

    do {
        max_diff = 0;  // максимальное изменение значений mat_U

        // Параллельный расчет с разбиением на горизонтальные полосы
        #pragma omp parallel for private(diff, max_diff_iter) shared(mat_U, mat_F, mat_U_new) reduction(max:max_diff)
        for (int i = 1; i < N - 1; i++) {
            max_diff_iter = 0;

            // Параллельная обработка строк
            for (int j = 1; j < N - 1; j++) {
                double temp = mat_U[i][j];
                mat_U_new[i][j] = 0.25 * (mat_U[i - 1][j] + mat_U[i + 1][j] + 
                                          mat_U[i][j - 1] + mat_U[i][j + 1] - h * h * mat_F[i][j]);
                diff = fabs(temp - mat_U_new[i][j]);
                if (max_diff_iter < diff) {
                    max_diff_iter = diff;
                }
            }

            // Обновление максимального изменения
            omp_set_lock(&dmax_lock);
            if (max_diff_iter > max_diff) {
                max_diff = max_diff_iter;
            }
            omp_unset_lock(&dmax_lock);
        }

        // Обновление значений в mat_U после завершения параллельных расчетов
        #pragma omp parallel for shared(mat_U, mat_U_new)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                mat_U[i][j] = mat_U_new[i][j];  // Обновление значений в исходном массиве
            }
        }

    } while (max_diff > EPS);  // Условие сходимости

    end_time = omp_get_wtime();

    printf("Elapsed time: %lf ms\n", (end_time - start_time) * 1000);

    free_memory();

    return 0;
}
