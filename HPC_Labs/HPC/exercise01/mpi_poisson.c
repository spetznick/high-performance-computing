/*
 * SEQ_Poisson.c
 * 2D Poison equation solver
 */

#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEBUG 0

#define max(a, b) ((a) > (b) ? a : b)

enum { X_DIR, Y_DIR };

/* global variables */
char DEBUG_FILENAME[40];
char input_filename[40];
int gridsize[2];
double precision_goal; /* precision_goal of solution */
int max_iter;          /* maximum number of iterations alowed */
MPI_Datatype border_type[2];
double omega = 1.0;

/* benchmark related variables */
clock_t ticks;    /* number of systemticks */
int timer_on = 0; /* is timer running? */

/* local grid related variables */
double **phi; /* grid */
int **source; /* TRUE if subgrid element is a source */
int dim[2];   /* grid dimensions */

/* timer variable */
int world_rank;
double wtime;
double time_spent_communication = 0;

/* process specific variables */
int proc_rank;     /* rank of current process */
int proc_coord[2]; /* coordinates of current process in processgrid */
int proc_top, proc_right, proc_bottom,
    proc_left; /* ranks of neigboring procs */
int offset[2];

int num_procs;      /* total number of processes */
int proc_dim[2];    /* process grid dimensions*/
MPI_Comm grid_comm; /* grid COMMUNICATOR */
MPI_Status status;

void time_communication(const char second_call);
void Setup_Args(int argc, char **argv);
void Setup_MPI_Datatypes();
void Exchange_Borders();
void Setup_Grid();
double Do_Step(int parity);
int Solve();
void Write_Grid();
void Clean_Up();
void Debug(char *mesg, int terminate);
void start_timer();
void resume_timer();
void stop_timer();
void print_timer();

void Setup_Args(int argc, char **argv) {
    /* Retrieve the number of processes */
    MPI_Comm_size(MPI_COMM_WORLD,
                  &num_procs); /* find out how many processes there are */
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    /* Calculate the number of processes per column and per row for the grid */
    if (argc > 2) {
        proc_dim[X_DIR] = atoi(argv[1]);
        proc_dim[Y_DIR] = atoi(argv[2]);
        if (proc_dim[X_DIR] * proc_dim[Y_DIR] != num_procs)
            Debug("ERROR Proces grid dimensions do not match with num_procs ",
                  1);
        strncpy(input_filename, "input.dat", 20);
    } else {
        Debug("ERROR Wrong parameter input", 1);
    }
    // optional parameters
    if (argc >= 4) {
        omega = atof(argv[3]);
    }
    if (argc >= 5) {
        strncpy(input_filename, argv[4], 20);
    }
}

void Setup_MPI_Datatypes() {
    // Debug("Setup_MPI_Datatypes", 0);
    /* Datatype for vertical data exchange (Y_DIR) */
    MPI_Type_vector(dim[X_DIR] - 2, 1, dim[Y_DIR], MPI_DOUBLE,
                    &border_type[Y_DIR]);
    MPI_Type_commit(&border_type[Y_DIR]);

    /* Datatype for horizontal data exchange (X_DIR) */
    MPI_Type_vector(dim[Y_DIR] - 2, 1, 1, MPI_DOUBLE, &border_type[X_DIR]);
    MPI_Type_commit(&border_type[X_DIR]);
}

void Exchange_Borders() {
    // Debug("Exchange_Borders", 0);
    MPI_Sendrecv(&phi[1][1], 1, border_type[Y_DIR], proc_top, 0,
                 &phi[1][dim[Y_DIR] - 1], 1, border_type[Y_DIR], proc_bottom, 0,
                 grid_comm, &status); /* all traffic in direction "top" */
    MPI_Sendrecv(&phi[1][dim[Y_DIR] - 2], 1, border_type[Y_DIR], proc_bottom, 0,
                 &phi[1][0], 1, border_type[Y_DIR], proc_top, 0, grid_comm,
                 &status); /* all traffic in direction "bottom" */
    MPI_Sendrecv(&phi[1][1], 1, border_type[X_DIR], proc_left, 0,
                 &phi[dim[X_DIR] - 1][1], 1, border_type[X_DIR], proc_right, 0,
                 grid_comm, &status); /* all traffic in direction "left" */
    MPI_Sendrecv(&phi[dim[X_DIR] - 2][1], 1, border_type[X_DIR], proc_right, 0,
                 &phi[0][1], 1, border_type[X_DIR], proc_left, 0, grid_comm,
                 &status); /* all traffic in direction "right" */
}

void start_timer() {
    if (!timer_on) {
        MPI_Barrier(MPI_COMM_WORLD);
        ticks = clock();
        wtime = MPI_Wtime();
        timer_on = 1;
    }
}

void resume_timer() {
    if (!timer_on) {
        ticks = clock() - ticks;
        wtime = MPI_Wtime() - wtime;
        timer_on = 1;
    }
}

void stop_timer() {
    if (timer_on) {
        ticks = clock() - ticks;
        wtime = MPI_Wtime() - wtime;
        timer_on = 0;
    }
}

void print_timer() {
    if (timer_on) {
        stop_timer();
        printf("(%i) Elapsed Wtime %14.6f s (%5.1f%% CPU)\n", proc_rank, wtime,
               100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
        resume_timer();
    } else
        printf("(%i) Elapsed Wtime %14.6f s (%5.1f%% CPU)\n", proc_rank, wtime,
               100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
}

void time_communication(const char second_call) {
    static double start_time;
    if (!second_call) {
        stop_timer();
        start_time = wtime;
        resume_timer();
    } else {
        stop_timer();
        time_spent_communication += wtime - start_time;
        resume_timer();
    }
}

void compute_communication_per_iteration(double runtime, int count) {
    printf("(%i) Time spent in communication: %14.6f s (%2.1f %%)\n", proc_rank,
           time_spent_communication, 100 * time_spent_communication / runtime);
    printf("(%i) Times per iteration: %14.6f s of %.6f s\n", proc_rank,
           time_spent_communication / (double)count, runtime / (double)count);
    time_spent_communication = 0.;
}

void Debug(char *mesg, int terminate) {
    if (DEBUG || terminate) printf("%s\n", mesg);
    if (terminate) exit(EXIT_FAILURE);
}

void Setup_Proc_Grid() {
    int wrap_around[2];
    int reorder;
    /* Create process topology (2D grid) */
    wrap_around[X_DIR] = 0;
    wrap_around[Y_DIR] = 0; /* do not connect first and last process */
    reorder = 0;            /* reorder process ranks */
    MPI_Cart_create(MPI_COMM_WORLD, 2, proc_dim, wrap_around, reorder,
                    &grid_comm); /* Creates a new communicator grid_comm */
    /* Retrieve new rank and cartesian coordinates of this process */
    MPI_Comm_rank(grid_comm,
                  &proc_rank); /* Rank of process in new communicator */
    MPI_Cart_coords(grid_comm, proc_rank, 2,
                    proc_coord); /* Coordinates of process in new communicator*/
    /* calculate ranks of neighboring processes */
    /* rank of processes proc_top and proc_bottom */
    MPI_Cart_shift(grid_comm, Y_DIR, 1, &proc_top, &proc_bottom);
    /* rank of processes proc_left and proc_right */
    MPI_Cart_shift(grid_comm, X_DIR, 1, &proc_left, &proc_right);
    /* rank of processes proc_left and proc_right */
}

void Setup_Grid() {
    int x, y, s;
    double source_x, source_y, source_val;
    FILE *f;
    int upper_offset[2];

    Debug("Setup_Subgrid", 0);
    if (proc_rank == 0) {
        f = fopen(input_filename, "r");
        if (f == NULL) {
            int result = snprintf(DEBUG_FILENAME, sizeof(DEBUG_FILENAME),
                                  "Error opening %s", input_filename);
            Debug(DEBUG_FILENAME, 1);
        }
        fscanf(f, "nx: %i\n", &gridsize[X_DIR]);
        fscanf(f, "ny: %i\n", &gridsize[Y_DIR]);
        fscanf(f, "precision goal: %lf\n", &precision_goal);
        fscanf(f, "max iterations: %i\n", &max_iter);
    }
    MPI_Bcast(gridsize, 2, MPI_INT, 0, grid_comm);
    MPI_Bcast(&precision_goal, 1, MPI_DOUBLE, 0, grid_comm);
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, grid_comm);

    /* Calculate top left corner coordinates of local grid */
    offset[X_DIR] = gridsize[X_DIR] * proc_coord[X_DIR] / proc_dim[X_DIR];
    offset[Y_DIR] = gridsize[Y_DIR] * proc_coord[Y_DIR] / proc_dim[Y_DIR];
    upper_offset[X_DIR] =
        gridsize[X_DIR] * (proc_coord[X_DIR] + 1) / proc_dim[X_DIR];
    upper_offset[Y_DIR] =
        gridsize[Y_DIR] * (proc_coord[Y_DIR] + 1) / proc_dim[Y_DIR];
    /* Calculate dimensions of local grid */
    dim[Y_DIR] = upper_offset[Y_DIR] - offset[Y_DIR];
    dim[X_DIR] = upper_offset[X_DIR] - offset[X_DIR];
    /* Add space for rows/columns of neighboring grid */
    dim[Y_DIR] += 2;
    dim[X_DIR] += 2;

    /* allocate memory */
    if ((phi = malloc(dim[X_DIR] * sizeof(*phi))) == NULL)
        Debug("Setup_Subgrid : malloc(phi) failed", 1);
    if ((source = malloc(dim[X_DIR] * sizeof(*source))) == NULL)
        Debug("Setup_Subgrid : malloc(source) failed", 1);
    if ((phi[0] = malloc(dim[Y_DIR] * dim[X_DIR] * sizeof(**phi))) == NULL)
        Debug("Setup_Subgrid : malloc(*phi) failed", 1);
    if ((source[0] = malloc(dim[Y_DIR] * dim[X_DIR] * sizeof(**source))) ==
        NULL)
        Debug("Setup_Subgrid : malloc(*source) failed", 1);
    for (x = 1; x < dim[X_DIR]; x++) {
        phi[x] = phi[0] + x * dim[Y_DIR];
        source[x] = source[0] + x * dim[Y_DIR];
    }

    /* set all values to '0' */
    for (x = 0; x < dim[X_DIR]; x++)
        for (y = 0; y < dim[Y_DIR]; y++) {
            phi[x][y] = 0.0;
            source[x][y] = 0;
        }

    /* put sources in field */
    do {
        if (proc_rank == 0) {
            s = fscanf(f, "source: %lf %lf %lf\n", &source_x, &source_y,
                       &source_val);
        }
        MPI_Bcast(&s, 1, MPI_INT, 0, grid_comm);
        if (s == 3) {
            MPI_Bcast(&source_x, 1, MPI_DOUBLE, 0, grid_comm);
            MPI_Bcast(&source_y, 1, MPI_DOUBLE, 0, grid_comm);
            MPI_Bcast(&source_val, 1, MPI_DOUBLE, 0, grid_comm);
            x = source_x * gridsize[X_DIR];
            y = source_y * gridsize[Y_DIR];
            x = x - offset[X_DIR] + 1;
            y = y - offset[Y_DIR] + 1;
            if (x > 0 && x < dim[X_DIR] - 1 && y > 0 &&
                y < dim[Y_DIR] - 1) { /* indices in domain of
                                         this process */
                phi[x][y] = source_val;
                source[x][y] = 1;
            }
        }

    } while (s == 3);
    if (proc_rank == 0) {
        fclose(f);
    }
    Debug("Setup_Subgrid end", 0);
}

double Do_Step(int parity) {
    int x, y;
    double old_phi;
    double max_err = 0.0;
    double cij;

    /* calculate interior of grid */
    for (x = 1; x < dim[X_DIR] - 1; x++)
        for (y = 1; y < dim[Y_DIR] - 1; y++)
            if ((x + offset[X_DIR] + y + offset[Y_DIR]) % 2 == parity &&
                source[x][y] != 1) {
                old_phi = phi[x][y];
                cij = 0.25 * (phi[x + 1][y] + phi[x - 1][y] + phi[x][y + 1] +
                              phi[x][y - 1]);
                phi[x][y] = omega * cij + (1 - omega) * old_phi;
                if (max_err < fabs(old_phi - phi[x][y])) {
                    max_err = fabs(old_phi - phi[x][y]);
                }
            }

    return max_err;
}

int Solve() {
    int count = 0;
    double delta;
    double delta1, delta2;
    double global_delta;

    Debug("Solve", 0);

    /* give global_delta a higher value then precision_goal */
    delta = 2 * precision_goal;
    global_delta = 2 * precision_goal;
    while (global_delta > precision_goal && count < max_iter &&
           delta < (DBL_MAX / 2)) {
        Debug("Do_Step 0", 0);
        delta1 = Do_Step(0);
        Exchange_Borders();

        Debug("Do_Step 1", 0);
        delta2 = Do_Step(1);
        Exchange_Borders();

        delta = max(delta1, delta2);
        if (!(count % 10)) {
            // time_communication(0);
            MPI_Allreduce(&delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX,
                          grid_comm);
            // if (proc_rank == 0) printf("(%i), e: %.4lf\n", count,
            // global_delta);
            // time_communication(1);
        }
        count++;
    }
    if (delta >= (DBL_MAX / 2)) {
        printf("Solution is diverging. Aborting\n");
        fflush(stdout);
        MPI_Abort(grid_comm, 1);
    }
    if (proc_rank == 0) {
        printf("Number of iterations: %i, omega: %.2f, grid size: %i\n", count,
               omega, gridsize[X_DIR]);
        printf("Delta: %f\n", global_delta);
    }
    return count;
}

void Write_Grid() {
    int x, y;
    FILE *f;
    char filename[40];
    sprintf(filename, "output%i.dat", proc_rank);

    if ((f = fopen(filename, "w")) == NULL) {
        Debug("Write_Grid : fopen failed", 1);
    }

    Debug("Write_Grid", 0);

    for (x = 1; x < dim[X_DIR] - 1; x++)
        for (y = 1; y < dim[Y_DIR] - 1; y++)
            fprintf(f, "%i %i %f\n", x + offset[X_DIR], y + offset[Y_DIR],
                    phi[x][y]);

    fclose(f);
}

void Merge_Grid_Files() {
    FILE *read_file, *write_file;
    char write_filename[40];
    char read_filename[40];
    sprintf(write_filename, "par_output.dat");
    int x_, y_;
    double phi_;

    if ((write_file = fopen(write_filename, "w")) == NULL) {
        Debug("Write_Grid : fopen failed", 1);
    }
    for (int proc_idx = 0; proc_idx < num_procs; proc_idx++) {
        snprintf(read_filename, sizeof(read_filename), "output%i.dat",
                 proc_idx);
        read_file = fopen(read_filename, "r");
        if (read_file == NULL) {
            int result = snprintf(DEBUG_FILENAME, sizeof(DEBUG_FILENAME),
                                  "Error opening %s", read_filename);
            Debug(DEBUG_FILENAME, 1);
        }
        while (fscanf(read_file, "%i %i %lf\n", &x_, &y_, &phi_) == 3) {
            fprintf(write_file, "%i %i %lf\n", x_, y_, phi_);
        }
        fclose(read_file);
    }
    fclose(write_file);
}

void Clean_Up() {
    Debug("Clean_Up", 0);

    free(phi[0]);
    free(phi);
    free(source[0]);
    free(source);
}

int main(int argc, char **argv) {
    // Initialize MPI, find out MPI communicator size and process
    // rank
    MPI_Init(&argc, &argv);
    Setup_Args(argc, argv);
    Setup_Proc_Grid();

    for (size_t i = 0; i < 1; i++) {
        Setup_Grid();
        Setup_MPI_Datatypes();
        start_timer();
        int count = Solve();
        print_timer();
        stop_timer();
        compute_communication_per_iteration(wtime, count);
    }
    Write_Grid();
    MPI_Barrier(grid_comm);
    if (proc_rank == 0) {
        Merge_Grid_Files();
    }
    Clean_Up();
    MPI_Finalize();
    return 0;
}
