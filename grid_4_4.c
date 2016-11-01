#include <mpi.h>
#include <stdio.h>
#define TAG 0
#define N 4
#define RHO 0.5 
#define ETA 0.0002
#define G 0.75

int main(int argc, char** argv) {
    int u1_matrix[N][N], u2_matrix[N][N];
    int number_iterations = atoi(argv[4]);

    u1_matrix[2][2] = 1;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    //Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int matrix_row_index = rank / 4;
    int matrix_column_index = rank % 4;
    int u1 = u1_matrix[matrix_row_index][matrix_column_index];
    int u2 = u2_matrix[matrix_row_index][matrix_column_index];

    MPI_Request request1, request2, request3, request4, request5,
    request6, request7, request8, request9, request10, request11, request12,
    request13, request14, request15, request16, request17, request18,
    request19, request20, request21, request22, request23, request24, request25;
   
    
    if (rank == 0) {
        int u1_1_0;
        MPI_Irecv(&u1_1_0, 1, MPI_INT, 4, TAG, MPI_COMM_WORLD, &request22);

        int u = G*u1_1_0;
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2;
    }

    if (rank == 1) {
        int u1_1_1;
        MPI_Isend(&u1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request5);
        MPI_Irecv(&u1_1_1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request1);

        int u = G*u1_1_1;
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2;        
    }

    if (rank == 2) {
        int u1_1_2;
        MPI_Isend(&u1, 1, MPI_INT, 3, TAG, MPI_COMM_WORLD, &request23);
        MPI_Isend(&u1, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request27);

        MPI_Irecv(&u1_1_2, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request9);

        int u = G*u1_1_2;
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2; 
    }

    if (rank == 3) {
        int u1_0_2;
        MPI_Irecv(&u1_0_2, 1, MPI_INT, 2, TAG, MPI_COMM_WORLD, &request23);

        int u = G*u1_0_2;
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2;
    }

    if (rank == 4) {
        int u1_1_1;
        MPI_Isend(&u1, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD, &request22);
        MPI_Isend(&u1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request6);

        MPI_Irecv(&u1_1_1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request2);

        int u = G*u1_1_1;
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2; 
    }

    if (rank == 5) {
        int u1_0_1, u1_1_0, u1_1_2, u1_2_1;
        MPI_Isend(&u1, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD, &request1);
        MPI_Isend(&u1, 1, MPI_INT, 4, TAG, MPI_COMM_WORLD, &request2);
        MPI_Isend(&u1, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request3);
        MPI_Isend(&u1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request4);        

        MPI_Irecv(&u1_0_1, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD, &request5);
        MPI_Irecv(&u1_1_0, 1, MPI_INT, 4, TAG, MPI_COMM_WORLD, &request6);
        MPI_Irecv(&u1_1_2, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request7);
        MPI_Irecv(&u1_2_1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request8);

        int u = (RHO*(u1_0_1 + u1_1_0 + u1_2_1 + u1_1_2 - (4*u1)) + 2*u1 - (1-ETA)*u2)/(1+ETA);
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2;
    }

    if (rank == 6) {
        int u1_0_2, u1_1_1, u1_1_3, u1_2_2;
        MPI_Isend(&u1, 1, MPI_INT, 2, TAG, MPI_COMM_WORLD, &request9);
        MPI_Isend(&u1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request7);
        MPI_Isend(&u1, 1, MPI_INT, 7, TAG, MPI_COMM_WORLD, &request10);
        MPI_Isend(&u1, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request11);

        MPI_Irecv(&u1_0_2, 1, MPI_INT, 2, TAG, MPI_COMM_WORLD, &request27);
        MPI_Irecv(&u1_1_1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request3);
        MPI_Irecv(&u1_1_3, 1, MPI_INT, 7, TAG, MPI_COMM_WORLD, &request12);
        MPI_Irecv(&u1_2_2, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request13);

        int u = (RHO*(u1_0_2 + u1_1_1 + u1_2_2 + u1_1_3 - (4*u1)) + 2*u1 - (1-ETA)*u2)/(1+ETA);
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2;
    }

    if (rank == 7) {
        int u1_1_2;
        MPI_Isend(&u1, 1, MPI_INT, 3, TAG, MPI_COMM_WORLD, &request23);
        MPI_Isend(&u1, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request12);

        MPI_Irecv(&u1_1_2, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request10);

        int u = G*u1_1_2;
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2; 
    }

    if (rank == 8) {
        int u1_2_1;
        MPI_Isend(&u1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request14);
        MPI_Isend(&u1, 1, MPI_INT, 12, TAG, MPI_COMM_WORLD, &request24);

        MPI_Irecv(&u1_2_1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request17);

        int u = G*u1_2_1;
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2; 
    }

    if (rank == 9) {
        int u1_1_1, u1_2_0, u1_2_2, u1_3_1;
        MPI_Isend(&u1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request8);
        MPI_Isend(&u1, 1, MPI_INT, 8, TAG, MPI_COMM_WORLD, &request17);
        MPI_Isend(&u1, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request18);
        MPI_Isend(&u1, 1, MPI_INT, 13, TAG, MPI_COMM_WORLD, &request19);

        MPI_Irecv(&u1_1_1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request4);
        MPI_Irecv(&u1_2_0, 1, MPI_INT, 8, TAG, MPI_COMM_WORLD, &request14);
        MPI_Irecv(&u1_2_2, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request15);
        MPI_Irecv(&u1_3_1, 1, MPI_INT, 13, TAG, MPI_COMM_WORLD, &request16);

        int u = (RHO*(u1_1_1 + u1_2_0 + u1_2_2 + u1_3_1 - (4*u1)) + 2*u1 - (1-ETA)*u2)/(1+ETA);
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2;
    }

    if (rank == 10) {
        int u1_1_2, u1_2_1, u1_2_3, u1_3_2;
        
        MPI_Isend(&u1, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request13);
        MPI_Isend(&u1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request15);
        MPI_Isend(&u1, 1, MPI_INT, 11, TAG, MPI_COMM_WORLD, &request22);
        MPI_Isend(&u1, 1, MPI_INT, 14, TAG, MPI_COMM_WORLD, &request16);

        MPI_Irecv(&u1_1_2, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request11);
        MPI_Irecv(&u1_2_1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request18);
        MPI_Irecv(&u1_2_3, 1, MPI_INT, 11, TAG, MPI_COMM_WORLD, &request20);
        MPI_Irecv(&u1_3_2, 1, MPI_INT, 14, TAG, MPI_COMM_WORLD, &request21);

        int u = (RHO*(u1_1_2 + u1_2_1 + u1_2_3 + u1_3_2 - (4*u1)) + 2*u1 - (1-ETA)*u2)/(1+ETA);
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2;
    }

    if (rank == 11) {
        int u1_2_2;
        MPI_Isend(&u1, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request20);
        MPI_Isend(&u1, 1, MPI_INT, 15, TAG, MPI_COMM_WORLD, &request25);

        MPI_Irecv(&u1_2_2, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request22);

        int u = G*u1_2_2;
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2;
    }

    if (rank == 12) {
        int u1_2_0;
        MPI_Irecv(&u1_2_0, 1, MPI_INT, 8, TAG, MPI_COMM_WORLD, &request24);

        int u = G*u1_2_0;
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2;
    }

    if (rank == 13) {
        int u1_2_1;
        MPI_Isend(&u1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request16);

        MPI_Irecv(&u1_2_1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request19);

        int u = G*u1_2_1;
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2;
    }

    if (rank == 14) {
        int u1_2_2;
        MPI_Isend(&u1, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request21);
        MPI_Isend(&u1, 1, MPI_INT, 15, TAG, MPI_COMM_WORLD, &request25);

        MPI_Irecv(&u1_2_2, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request21);
        
        int u = G*u1_2_2;
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2;
    }

    if (rank == 15) {
        int u1_3_2;
        MPI_Irecv(&u1_3_2, 1, MPI_INT, 14, TAG, MPI_COMM_WORLD, &request25);

        int u = G*u1_3_2;
        u2 = u1;
        u1 = u;
        u1_matrix[matrix_row_index][matrix_column_index] = u1;
        u2_matrix[matrix_row_index][matrix_column_index] = u2;        
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}
