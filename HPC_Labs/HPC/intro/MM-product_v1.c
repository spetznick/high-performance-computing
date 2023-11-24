/******************************************************************************
 * FILE: mm.c
 * DESCRIPTION:
 *   This program calculates the product of matrix a[N][N] and b[N][N],
 *   the result is stored in matrix c[N][N].
 *   The max dimension of the matrix is constraint with static array
 *declaration, for a larger matrix you may consider dynamic allocation of the
 *arrays, but it makes a parallel code much more complicated (think of
 *communication), so this is only optional.
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

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10

int main(int argc, char *argv[]) {
    // printf("Matrix Multiplication \n");

    // Variables for the process rank and number of processes
    int rank, numProcs;
    int count = 0;
    int stripsize, b_row, b_col;
    // MPI_Datatype strip;
    int blocksize, i, j, k;
    // MPI_Status status;
    // Initialize MPI, find out MPI communicator size and process rank
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (numProcs != N) {
        printf("This only works for %d processes, started with %d", N,
               numProcs);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // int stride = 0;
    // blocksize = (int) ceil((float) N/numProcs);

    double a[N * N];       /* matrix A to be multiplied */
    double b[N * N];       /* matrix B to be multiplied */
    double c[N * N];       /* result matrix C */
    double c_local[N * N]; /* result matrix C */
    double c_check[N * N]; /* matrix C for test purposes */

    // MPI_Type_vector(N / numProcs, b_row, b_col, MPI_DOUBLE, &strip);
    // MPI_Type_commit(&strip);

    /* for simplicity, set N=N=N=N  */
    if (rank == 0) {
        /*** Initialize matrices ***/
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                a[i * N + j] = i + j;
            }
        }

        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                b[i * N + j] = i * j;
            }
        }

        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                c[i * N + j] = 0;
                c_check[i * N + j] = 0;
            }
        }

        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < N; k++) {
                    c_check[i * N + j] +=
                        a[i * N + k] *
                        b[j * N + k];  // matrix stores in row major, so j and k
                                       // are switched
                }
            }
        }
        // printf("comment %d, rank %d\n", count++, rank);
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            c_local[i * N + j] = 0;
        }
    }
    // printf("comment %d, rank %d\n", count++, rank);

    MPI_Bcast(a, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // cast row of B which stores a column of B
    // MPI_Scatter(&b, blocksize * N, MPI_FLOAT, row, blksz * N, MPI_FLOAT,
    // 0,MPI_COMM_WORLD);
    MPI_Bcast(b, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // MPI_Bcast(&c, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // printf("comment %d, rank %d\n", count++, rank);

    for (i = 0; i < N; i++) {
        for (k = 0; k < N; k++) {
            c_local[i * N + rank] += a[i * N + k] * b[rank * N + k];
        }
    }
    // printf("comment %d, rank %d\n", count++, rank);

    // printf("node %d, matrix c[0*N+%d]=%f\n", rank, rank, c_local[0 * N +
    // rank]);

    /* Parallelize the computation of the following matrix-matrix
      multiplication. How to partition and distribute the initial matrices, the
      work, and collecting final results.
    */

    MPI_Reduce(c_local, c, N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (i = 0; i < N; i++) {
            printf("c[%d][%d]=%.1f and c_check[%d][%d]=%.1f\n", i, i,
                   c[i * N + i], i, i, c_check[i * N + i]);
            if (c[i * N + i] != c_check[i * N + i]) {
                printf("Result wrong");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    }
    /*  perform time measurement. Always check the correctness of the parallel
       results by printing a few values of c[i][j] and compare with the
       sequential output.
    */

    // MPI_Type_free(&strip);
    MPI_Finalize();
    return 0;
}
