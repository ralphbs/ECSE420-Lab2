    #include <mpi.h>
    #include <stdio.h>
    #define TAG 0
    #define N 4
    #define RHO 0.5 
    #define ETA 0.0002
    #define G 0.75

    int main(int argc, char** argv) {
        int u_matrix[N][N], u1_matrix[N][N], u2_matrix[N][N];
        int number_iterations = atoi(argv[4]);

        u1_matrix[2][2] = 1;
        int i;
        for (i = 0; i<number_iterations; i++) {
        // Initialize the MPI environment
            MPI_Init(&argc, &argv);

            //Get the number of processes
            int size;
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            // Get the rank of the processes
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            int matrix_row_index = rank / 4;
            int matrix_column_index = rank % 4;
            int u1 = u1_matrix[matrix_row_index][matrix_column_index];
            int u2 = u2_matrix[matrix_row_index][matrix_column_index];

            MPI_Status status;
            MPI_Request request[25];
       
        
        if (rank == 0) {
            int u1_1_0;
            MPI_Irecv(&u1_1_0, 1, MPI_INT, 4, TAG, MPI_COMM_WORLD, &request[22]);
            MPI_Wait(&request[22], &status);

            int u = G*u1_1_0;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }

        if (rank == 1) {
            int u1_1_1;
            MPI_Isend(&u1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request[5]);
            MPI_Wait(&request[5], &status);
            
            MPI_Irecv(&u1_1_1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request[1]);
            MPI_Wait(&request[1], &status);

            int u = G*u1_1_1;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;        
        }

        if (rank == 2) {
            int u1_1_2;
            MPI_Isend(&u1, 1, MPI_INT, 3, TAG, MPI_COMM_WORLD, &request[23]);
            MPI_Wait(&request[23], &status);

            MPI_Isend(&u1, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request[27]);
            MPI_Wait(&request[27], &status);

            MPI_Irecv(&u1_1_2, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request[9]);
            MPI_Wait(&request[9], &status);

            int u = G*u1_1_2;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2; 
        }

        if (rank == 3) {
            int u1_0_2;
            MPI_Irecv(&u1_0_2, 1, MPI_INT, 2, TAG, MPI_COMM_WORLD, &request[23]);
            MPI_Wait(&request[23], &status);

            int u = G*u1_0_2;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }

        if (rank == 4) {
            int u1_1_1;
            MPI_Isend(&u1, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD, &request[22]);
            MPI_Wait(&request[22], &status);

            MPI_Isend(&u1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request[6]);
            MPI_Wait(&request[6], &status);

            MPI_Irecv(&u1_1_1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request[2]);
            MPI_Wait(&request[2], &status);

            int u = G*u1_1_1;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2; 
        }

        if (rank == 5) {
            int u1_0_1, u1_1_0, u1_1_2, u1_2_1;
            MPI_Isend(&u1, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD, &request[1]);
            MPI_Wait(&request[1], &status);

            MPI_Isend(&u1, 1, MPI_INT, 4, TAG, MPI_COMM_WORLD, &request[2]);
            MPI_Wait(&request[2], &status);

            MPI_Isend(&u1, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request[3]);
            MPI_Wait(&request[3], &status);

            MPI_Isend(&u1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request[4]);
            MPI_Wait(&request[4], &status);        

            MPI_Irecv(&u1_0_1, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD, &request[5]);
            MPI_Wait(&request[5], &status);

            MPI_Irecv(&u1_1_0, 1, MPI_INT, 4, TAG, MPI_COMM_WORLD, &request[6]);
            MPI_Wait(&request[6], &status);
            
            MPI_Irecv(&u1_1_2, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request[7]);
            MPI_Wait(&request[7], &status);
            
            MPI_Irecv(&u1_2_1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request[8]);
            MPI_Wait(&request[8], &status);

            int u = (RHO*(u1_0_1 + u1_1_0 + u1_2_1 + u1_1_2 - (4*u1)) + 2*u1 - (1-ETA)*u2)/(1+ETA);
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }

        if (rank == 6) {
            int u1_0_2, u1_1_1, u1_1_3, u1_2_2;
            MPI_Isend(&u1, 1, MPI_INT, 2, TAG, MPI_COMM_WORLD, &request[9]);
            MPI_Wait(&request[9], &status);

            MPI_Isend(&u1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request[7]);
            MPI_Wait(&request[7], &status);
            
            MPI_Isend(&u1, 1, MPI_INT, 7, TAG, MPI_COMM_WORLD, &request[10]);
            MPI_Wait(&request[10], &status);
            
            MPI_Isend(&u1, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request[11]);
            MPI_Wait(&request[11], &status);

            MPI_Irecv(&u1_0_2, 1, MPI_INT, 2, TAG, MPI_COMM_WORLD, &request[27]);
            MPI_Wait(&request[27], &status);
           
            MPI_Irecv(&u1_1_1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request[3]);
            MPI_Wait(&request[3], &status);

            MPI_Irecv(&u1_1_3, 1, MPI_INT, 7, TAG, MPI_COMM_WORLD, &request[12]);
            MPI_Wait(&request[12], &status);

            MPI_Irecv(&u1_2_2, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request[13]);
            MPI_Wait(&request[13], &status);

            int u = (RHO*(u1_0_2 + u1_1_1 + u1_2_2 + u1_1_3 - (4*u1)) + 2*u1 - (1-ETA)*u2)/(1+ETA);
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }

        if (rank == 7) {
            int u1_1_2;
            //MPI_Isend(&u1, 1, MPI_INT, 3, TAG, MPI_COMM_WORLD, &request23);
            MPI_Isend(&u1, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request[12]);
            MPI_Wait(&request[12], &status);

            MPI_Irecv(&u1_1_2, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request[10]);
            MPI_Wait(&request[10], &status);

            int u = G*u1_1_2;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2; 
        }

        if (rank == 8) {
            int u1_2_1;
            MPI_Isend(&u1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request[14]);
            MPI_Wait(&request[14], &status);
           
            MPI_Isend(&u1, 1, MPI_INT, 12, TAG, MPI_COMM_WORLD, &request[24]);
            MPI_Wait(&request[24], &status);

            MPI_Irecv(&u1_2_1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request[17]);
            MPI_Wait(&request[17], &status);

            int u = G*u1_2_1;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2; 
        }

        if (rank == 9) {
            int u1_1_1, u1_2_0, u1_2_2, u1_3_1;
            MPI_Isend(&u1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request[8]);
            MPI_Wait(&request[8], &status);

            MPI_Isend(&u1, 1, MPI_INT, 8, TAG, MPI_COMM_WORLD, &request[17]);
            MPI_Wait(&request[17], &status);

            MPI_Isend(&u1, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request[18]);
            MPI_Wait(&request[18], &status);
            
            MPI_Isend(&u1, 1, MPI_INT, 13, TAG, MPI_COMM_WORLD, &request[19]);
            MPI_Wait(&request[19], &status);

            MPI_Irecv(&u1_1_1, 1, MPI_INT, 5, TAG, MPI_COMM_WORLD, &request[4]);
            MPI_Wait(&request[4], &status);

            MPI_Irecv(&u1_2_0, 1, MPI_INT, 8, TAG, MPI_COMM_WORLD, &request[14]);
            MPI_Wait(&request[14], &status);

            
            MPI_Irecv(&u1_2_2, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request[15]);
            MPI_Wait(&request[15], &status);
            
            MPI_Irecv(&u1_3_1, 1, MPI_INT, 13, TAG, MPI_COMM_WORLD, &request[16]);
            MPI_Wait(&request[16], &status);


            int u = (RHO*(u1_1_1 + u1_2_0 + u1_2_2 + u1_3_1 - (4*u1)) + 2*u1 - (1-ETA)*u2)/(1+ETA);
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }

        if (rank == 10) {
            int u1_1_2, u1_2_1, u1_2_3, u1_3_2;
            
            MPI_Isend(&u1, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request[13]);
            MPI_Wait(&request[13], &status);

            MPI_Isend(&u1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request[15]);
            MPI_Wait(&request[15], &status);
            
            MPI_Isend(&u1, 1, MPI_INT, 11, TAG, MPI_COMM_WORLD, &request[22]);
            MPI_Wait(&request[22], &status);

            MPI_Isend(&u1, 1, MPI_INT, 14, TAG, MPI_COMM_WORLD, &request[16]);
            MPI_Wait(&request[16], &status);

            MPI_Irecv(&u1_1_2, 1, MPI_INT, 6, TAG, MPI_COMM_WORLD, &request[11]);
            MPI_Wait(&request[11], &status);

            MPI_Irecv(&u1_2_1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request[18]);
            MPI_Wait(&request[18], &status);

            MPI_Irecv(&u1_2_3, 1, MPI_INT, 11, TAG, MPI_COMM_WORLD, &request[20]);
            MPI_Wait(&request[20], &status);
            
            MPI_Irecv(&u1_3_2, 1, MPI_INT, 14, TAG, MPI_COMM_WORLD, &request[21]);
            MPI_Wait(&request[21], &status);

            int u = (RHO*(u1_1_2 + u1_2_1 + u1_2_3 + u1_3_2 - (4*u1)) + 2*u1 - (1-ETA)*u2)/(1+ETA);
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }

        if (rank == 11) {
            int u1_2_2;
            MPI_Isend(&u1, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request[20]);
            MPI_Wait(&request[20], &status);
            //MPI_Isend(&u1, 1, MPI_INT, 15, TAG, MPI_COMM_WORLD, &request25);

            MPI_Irecv(&u1_2_2, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request[22]);
            MPI_Wait(&request[22], &status);

            int u = G*u1_2_2;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }

        if (rank == 12) {
            int u1_2_0;
            MPI_Irecv(&u1_2_0, 1, MPI_INT, 8, TAG, MPI_COMM_WORLD, &request[24]);
            MPI_Wait(&request[24], &status);

            int u = G*u1_2_0;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }

        if (rank == 13) {
            int u1_2_1;
            MPI_Isend(&u1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request[16]);
            MPI_Wait(&request[16], &status);


            MPI_Irecv(&u1_2_1, 1, MPI_INT, 9, TAG, MPI_COMM_WORLD, &request[19]);
            MPI_Wait(&request[19], &status);

            int u = G*u1_2_1;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }

        if (rank == 14) {
            int u1_2_2;
            MPI_Isend(&u1, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request[21]);
            MPI_Wait(&request[21], &status);

            
            MPI_Isend(&u1, 1, MPI_INT, 15, TAG, MPI_COMM_WORLD, &request[25]);
            MPI_Wait(&request[25], &status);

            MPI_Irecv(&u1_2_2, 1, MPI_INT, 10, TAG, MPI_COMM_WORLD, &request[16]);
            MPI_Wait(&request[16], &status);
            
            int u = G*u1_2_2;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }

        if (rank == 15) {
            int u1_3_2;
            MPI_Irecv(&u1_3_2, 1, MPI_INT, 14, TAG, MPI_COMM_WORLD, &request[25]);
            MPI_Wait(&request[25], &status);

            int u = G*u1_3_2;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;        
        }

        // Finalize the MPI environment.
        MPI_Finalize();
        }
        for(int j = 0; j<4; j++){
            for(int k = 0; k<4; k++){
                printf("%d ", u_matrix[j][k]);
            }
            printf("\n");
        }        
    }
