#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MAX_MSIZE 10000
#define MSIZE     500

float **allocate(int size) {
    int i;
    float **m;
    m = malloc(sizeof(float *) * size);
    m[0] = malloc(sizeof(float) * size * size);
    for (i = 1; i < size; i++)
        m[i] = m[i - 1] + size;
    return m;
}

void initialize(float **a, float **b, int size) {
    int i, j;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            a[i][j] = (float)rand() / RAND_MAX;
            b[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

float **multiply(float **a, float **b, int size, int local_size) {
    float **c;
    c = allocate(size);

    // Realizar la multiplicación de matrices localmente
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < local_size; j++) {
            c[i][j] = 0.0;
            for (int k = 0; k < size; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

int main(int argc, char* argv[]) {
    int size, rank;
    float **a;
    float **b;
    float **c;
    double start_time, end_time, total_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_size = MSIZE / size;
    a = allocate(MSIZE);
    b = allocate(MSIZE);
    c = allocate(MSIZE);

    if (rank == 0) {
        initialize(a, b, MSIZE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Distribuir las matrices a y b a todos los procesos
    MPI_Bcast(a[0], MSIZE * MSIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b[0], MSIZE * MSIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Multiplicación de matrices localmente
    c = multiply(a, b, MSIZE, local_size);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    total_time = end_time - start_time;

    if (rank == 0) {
        printf("c[0][0] = %f\n", c[0][0]);
        printf("Tiempo total: %f segundos\n", total_time);
    }

    MPI_Finalize();

    free(a[0]); free(b[0]); free(c[0]);
    free(a); free(b); free(c);

    return 0;
}