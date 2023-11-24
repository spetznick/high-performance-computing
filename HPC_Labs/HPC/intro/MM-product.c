/******************************************************************************
 * FILE: mm.c
 * DESCRIPTION:
 *   This program calculates the product of matrix a[n][n] and b[n][n],
 *   the result is stored in matrix c[n][n].
 *   The max dimension of the matrix is constraint with static array
 *declaration, for a larger matrix you may consider dynamic allocation of the
 *arrays, but it makes a parallel code much more complicated (think of
 *communication), so this is only optional.
 * This code works under the assumption that the number of processes is less
 *than half the number of rows of a given matrix
 *
 ******************************************************************************/

// Plan for iterative solution:
//
// 1. communicate both matrices to all processes, number of processes matches
// number of rows
// 2. communicate A completely and the row from B that is required by the
// process
// 3. communicate partially such that the algorithm works for any number of
// processes

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// #define n 1000

void printMyElements(const double *arr, int n, int rank, int blocksize) {
    printf("rank: %d\n", rank);
    for (int j = 0; j < blocksize; j++) {
        for (int i = 0; i < n; i++) {
            printf("%.1f, ", arr[n * j + i]);
        }
        printf(";\n");
    }
}

char parallel_matrix_calc(int n, int rank, int numProcs) {
    int count = 0;
    int blocksize, i, j, k, lastBlocksize;
    double startTime = MPI_Wtime();

    int displs[numProcs];
    int counts[numProcs];
    blocksize = (int)floor((float)n / numProcs);
    lastBlocksize = n - blocksize * (numProcs - 1);
    for (i = 0; i < numProcs; i++) {
        if (i == (numProcs - 1)) {
            counts[i] = n * lastBlocksize;
        } else {
            counts[i] = n * blocksize;
        }
        displs[i] = n * blocksize * i;
    }
    if (rank == (numProcs - 1)) {
        blocksize = lastBlocksize;
    }

    double *a = malloc(sizeof(double) * n * n); /* matrix A to be multiplied */
    double *b = malloc(sizeof(double) * n * n); /* matrix B to be multiplied */
    double *b_local = malloc(sizeof(double) * n * blocksize); /* rows of B to be multiplied */
    double *c = malloc(sizeof(double) * n * n); /* result matrix C */
    double *c_local = malloc(sizeof(double) * n * blocksize); /* local matrix C */
    double *c_check = malloc(sizeof(double) * n * n); /* matrix C check */

    // /* for simplicity, set n=n=n=n  */
    if (rank == 0) {
        /*** Initialize matrices ***/
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                a[i * n + j] = i + j;
            }
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                b[i * n + j] = i * j;
            }
        }
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            c[i * n + j] = 0;
        }
    }

    for (i = 0; i < blocksize; i++) {
        for (j = 0; j < n; j++) {
            b_local[i * n + j] = 0;
        }
    }

    for (i = 0; i < blocksize; i++) {
        for (j = 0; j < n; j++) {
            c_local[i * n + j] = 0;
        }
    }

    MPI_Bcast(a, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, counts, displs, MPI_DOUBLE, b_local, counts[rank],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (j = 0; j < blocksize; j++) {
        for (i = 0; i < n; i++) {
            for (k = 0; k < n; k++) {
                // row of c
                c_local[j * n + i] += a[i * n + k] * b_local[j * n + k];
            }
        }
    }

    // received matrix is transposed!
    MPI_Gatherv(c_local, counts[rank], MPI_DOUBLE, c, counts, displs,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double endTime = MPI_Wtime();
    if (rank == 0) {
        for (i = 0; i < n; i++) {
            for ( j = 0; j < n; j++)
            {
                c_check[i * n + j] = 0;
            }
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                for (k = 0; k < n; k++) {
                    c_check[i * n + j] +=
                        a[i * n + k] *
                        b[k * n + j];
                }
            }
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++)
            {
                if (c[j * n + i] != c_check[i * n + j]) {
                    printf("Result wrong\n");
                    printf("c[%d][%d]=%.1f and c_check[%d][%d]=%.1f\n", i, j,
                        c[j * n + i], i, j, c_check[i * n + j]);
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
            }
        }
        printf("%d, %d, %.5f\n", n, numProcs,
               endTime - startTime);
    }

    free(a);
    free(b);
    free(b_local);
    free(c);
    free(c_local);
    free(c_check);
    return 0;
}

int main(int argc, char *argv[]) {
    // Variables for the process rank and number of processes
    int rank, numProcs;
    int n = 10;
    char return_val;

    // Initialize MPI, find out MPI communicator size and process rank
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("size, num procs, time\n");
    for (int i = 0; i < 31; i++) {
        return_val = parallel_matrix_calc(n, rank, numProcs);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
