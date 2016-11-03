#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#define TAG 0
#define N 4
#define RHO 0.5
#define ETA 0.0002
#define G 0.75

int main(int argc, char** argv) {
    // declare 3 matrices of single precision to store values at each node
    float u_matrix[4][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
    float u1_matrix[4][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
    float u2_matrix[4][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
    
    // access the argument specifiy number of iterations
    int number_iterations = atoi(argv[1]);
    
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    
    //Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get the rank of the processes
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Fetch the row index and the column index using the rank
    int matrix_row_index = rank / 4;
    int matrix_column_index = rank % 4;
    
    // Access matrices values u1 and u2 for current running process
    float u1 = u1_matrix[matrix_row_index][matrix_column_index];
    float u2 = u2_matrix[matrix_row_index][matrix_column_index];
    
    // Repeat the process according to specified number of iterations
    for(int i = 1; i<=number_iterations; i++){
        MPI_Status status[30];
        MPI_Request request[30];
        
        // if the process is a process running on an interior node i.e on either of the ranks 5,6,9,19
        if(rank == 5 || rank == 6 || rank == 9 || rank == 10){
            if (rank == 5) {
                float u1_0_1, u1_1_0, u1_1_2, u1_2_1;
                
                // send values of u1 to the right node and the bottom node(the interior nodes accessible to rank 5)
                MPI_Isend(&u1, 1, MPI_FLOAT, 6, TAG, MPI_COMM_WORLD, &request[3]);
                MPI_Isend(&u1, 1, MPI_FLOAT, 9, TAG, MPI_COMM_WORLD, &request[4]);
                
                // receive values from top, left, right, bottom
                MPI_Irecv(&u1_0_1, 1, MPI_FLOAT, 1, TAG, MPI_COMM_WORLD, &request[5]);
                MPI_Irecv(&u1_1_0, 1, MPI_FLOAT, 4, TAG, MPI_COMM_WORLD, &request[6]);
                MPI_Irecv(&u1_1_2, 1, MPI_FLOAT, 6, TAG, MPI_COMM_WORLD, &request[7]);
                MPI_Irecv(&u1_2_1, 1, MPI_FLOAT, 9, TAG, MPI_COMM_WORLD, &request[8]);
                
                // wait to receive values from top, left, right, bottom - this is a blocking call
                MPI_Wait(&request[5], &status[5]);
                MPI_Wait(&request[6], &status[6]);
                MPI_Wait(&request[7], &status[7]);
                MPI_Wait(&request[8], &status[8]);
                
                // upon receipt of the values, compute u using the provided equation in the assignment
                float u = (RHO*(u1_0_1 + u1_1_0 + u1_2_1 + u1_1_2 - (4*u1)) + 2*u1 - (1-ETA)*u2)/(1+ETA);
                // update pervious to become previous previous
                // and update current to become previous
                u2 = u1;
                u1 = u;
                // update the values in the matrices
                u_matrix[matrix_row_index][matrix_column_index] = u;
                u1_matrix[matrix_row_index][matrix_column_index] = u1;
                u2_matrix[matrix_row_index][matrix_column_index] = u2;
                
                // after the computation, send the updated value u1 to process 1 and 4
                MPI_Isend(&u1, 1, MPI_FLOAT, 1, TAG, MPI_COMM_WORLD, &request[1]);
                MPI_Isend(&u1, 1, MPI_FLOAT, 4, TAG, MPI_COMM_WORLD, &request[2]);
                
                MPI_Wait(&request[1], &status[1]);
                MPI_Wait(&request[2], &status[2]);
            }
            // Same logic applies for 6,9, and 10
            if (rank == 6) {
                float u1_0_2, u1_1_1, u1_1_3, u1_2_2;
                
                MPI_Isend(&u1, 1, MPI_FLOAT, 5, TAG, MPI_COMM_WORLD, &request[7]);
                MPI_Isend(&u1, 1, MPI_FLOAT, 10, TAG, MPI_COMM_WORLD, &request[11]);
                
                MPI_Irecv(&u1_0_2, 1, MPI_FLOAT, 2, TAG, MPI_COMM_WORLD, &request[27]);
                MPI_Irecv(&u1_1_1, 1, MPI_FLOAT, 5, TAG, MPI_COMM_WORLD, &request[3]);
                MPI_Irecv(&u1_1_3, 1, MPI_FLOAT, 7, TAG, MPI_COMM_WORLD, &request[12]);
                MPI_Irecv(&u1_2_2, 1, MPI_FLOAT, 10, TAG, MPI_COMM_WORLD, &request[13]);
                
                MPI_Wait(&request[27], &status[27]);
                MPI_Wait(&request[3], &status[3]);
                MPI_Wait(&request[12], &status[12]);
                MPI_Wait(&request[13], &status[13]);
                
                float u = (RHO*(u1_0_2 + u1_1_1 + u1_2_2 + u1_1_3 - (4*u1)) + 2*u1 - (1-ETA)*u2)/(1+ETA);
                u2 = u1;
                u1 = u;
                u_matrix[matrix_row_index][matrix_column_index] = u;
                u1_matrix[matrix_row_index][matrix_column_index] = u1;
                u2_matrix[matrix_row_index][matrix_column_index] = u2;
                
                MPI_Isend(&u1, 1, MPI_FLOAT, 2, TAG, MPI_COMM_WORLD, &request[9]);
                MPI_Isend(&u1, 1, MPI_FLOAT, 7, TAG, MPI_COMM_WORLD, &request[10]);
                
                MPI_Wait(&request[9], &status[9]);
                MPI_Wait(&request[10], &status[10]);
            }
            if(rank == 9){
                float u1_1_1, u1_2_0, u1_2_2, u1_3_1;
                MPI_Isend(&u1, 1, MPI_FLOAT, 5, TAG, MPI_COMM_WORLD, &request[8]);
                MPI_Isend(&u1, 1, MPI_FLOAT, 10, TAG, MPI_COMM_WORLD, &request[18]);
                
                MPI_Irecv(&u1_1_1, 1, MPI_FLOAT, 5, TAG, MPI_COMM_WORLD, &request[4]);
                MPI_Irecv(&u1_2_0, 1, MPI_FLOAT, 8, TAG, MPI_COMM_WORLD, &request[14]);
                MPI_Irecv(&u1_2_2, 1, MPI_FLOAT, 10, TAG, MPI_COMM_WORLD, &request[15]);
                MPI_Irecv(&u1_3_1, 1, MPI_FLOAT, 13, TAG, MPI_COMM_WORLD, &request[16]);
                
                MPI_Wait(&request[4], &status[4]);
                MPI_Wait(&request[14], &status[14]);
                MPI_Wait(&request[15], &status[15]);
                MPI_Wait(&request[16], &status[16]);
                
                float u = (RHO*(u1_1_1 + u1_2_0 + u1_2_2 + u1_3_1 - (4*u1)) + 2*u1 - (1-ETA)*u2)/(1+ETA);
                u2 = u1;
                u1 = u;
                u_matrix[matrix_row_index][matrix_column_index] = u;
                u1_matrix[matrix_row_index][matrix_column_index] = u1;
                u2_matrix[matrix_row_index][matrix_column_index] = u2;
                
                
                MPI_Isend(&u1, 1, MPI_FLOAT, 8, TAG, MPI_COMM_WORLD, &request[17]);
                MPI_Isend(&u1, 1, MPI_FLOAT, 13, TAG, MPI_COMM_WORLD, &request[19]);
                
                MPI_Wait(&request[17], &status[17]);
                MPI_Wait(&request[19], &status[19]);
            }
            if (rank == 10) {
                float u1_1_2, u1_2_1, u1_2_3, u1_3_2;
                
                MPI_Isend(&u1, 1, MPI_FLOAT, 6, TAG, MPI_COMM_WORLD, &request[13]);
                MPI_Isend(&u1, 1, MPI_FLOAT, 9, TAG, MPI_COMM_WORLD, &request[15]);
                
                MPI_Irecv(&u1_1_2, 1, MPI_FLOAT, 6, TAG, MPI_COMM_WORLD, &request[11]);
                MPI_Irecv(&u1_2_1, 1, MPI_FLOAT, 9, TAG, MPI_COMM_WORLD, &request[18]);
                MPI_Irecv(&u1_2_3, 1, MPI_FLOAT, 11, TAG, MPI_COMM_WORLD, &request[20]);
                MPI_Irecv(&u1_3_2, 1, MPI_FLOAT, 14, TAG, MPI_COMM_WORLD, &request[21]);
                
                MPI_Wait(&request[11], &status[11]);
                MPI_Wait(&request[18], &status[18]);
                MPI_Wait(&request[20], &status[20]);
                MPI_Wait(&request[21], &status[21]);
                
                float u = (RHO*(u1_1_2 + u1_2_1 + u1_2_3 + u1_3_2 - (4*u1)) + 2*u1 - (1-ETA)*u2)/(1+ETA);
                u2 = u1;
                u1 = u;
                u_matrix[matrix_row_index][matrix_column_index] = u;
                u1_matrix[matrix_row_index][matrix_column_index] = u1;
                u2_matrix[matrix_row_index][matrix_column_index] = u2;
                
                MPI_Isend(&u1, 1, MPI_FLOAT, 11, TAG, MPI_COMM_WORLD, &request[22]);
                MPI_Isend(&u1, 1, MPI_FLOAT, 14, TAG, MPI_COMM_WORLD, &request[16]);
                MPI_Wait(&request[22], &status[22]);
                MPI_Wait(&request[16], &status[16]);
                printf("The result of u in iteration %d is %lf\n", i, u);
            }
        }
        
        // For the boundary nodes, same logic applies
        // However the sending and receiveing is a bit more conservative as the assignment specifies
        if(rank == 1 || rank == 2 || rank == 4 || rank == 7 || rank == 8 || rank == 11 || rank == 13 || rank == 14){
            if (rank == 1) {
                float u1_1_1;
                // send the value of u1 to process 5
                MPI_Isend(&u1, 1, MPI_FLOAT, 5, TAG, MPI_COMM_WORLD, &request[5]);
                
                // wait until the value of u1_1_1 is received from process 5
                MPI_Irecv(&u1_1_1, 1, MPI_FLOAT, 5, TAG, MPI_COMM_WORLD, &request[1]);
                // block until the value has been received by process 1
                MPI_Wait(&request[1], &status[1]);
                
                // Upon receipt, compute the value of u and do necessary updates
                float u = G*u1_1_1;
                u2 = u1;
                u1 = u;
                u_matrix[matrix_row_index][matrix_column_index] = u;
                u1_matrix[matrix_row_index][matrix_column_index] = u1;
                u2_matrix[matrix_row_index][matrix_column_index] = u2;
            }
            // Same logic applies for 2, 4, 7, 8, 11, 13, 14
            if (rank == 2) {
                float u1_1_2;
                
                MPI_Isend(&u1, 1, MPI_FLOAT, 6, TAG, MPI_COMM_WORLD, &request[27]);
                
                MPI_Irecv(&u1_1_2, 1, MPI_FLOAT, 6, TAG, MPI_COMM_WORLD, &request[9]);
                MPI_Wait(&request[9], &status[9]);
                
                float u = G*u1_1_2;
                u2 = u1;
                u1 = u;
                u_matrix[matrix_row_index][matrix_column_index] = u;
                u1_matrix[matrix_row_index][matrix_column_index] = u1;
                u2_matrix[matrix_row_index][matrix_column_index] = u2;
                
                MPI_Isend(&u1, 1, MPI_FLOAT, 3, TAG, MPI_COMM_WORLD, &request[23]);
                MPI_Wait(&request[23], &status[23]);
            }
            if (rank == 4) {
                float u1_1_1;
                
                MPI_Isend(&u1, 1, MPI_FLOAT, 5, TAG, MPI_COMM_WORLD, &request[6]);
                
                MPI_Irecv(&u1_1_1, 1, MPI_FLOAT, 5, TAG, MPI_COMM_WORLD, &request[2]);
                MPI_Wait(&request[2], &status[2]);
                
                float u = G*u1_1_1;
                u2 = u1;
                u1 = u;
                u_matrix[matrix_row_index][matrix_column_index] = u;
                u1_matrix[matrix_row_index][matrix_column_index] = u1;
                u2_matrix[matrix_row_index][matrix_column_index] = u2;
                
                MPI_Isend(&u1, 1, MPI_FLOAT, 0, TAG, MPI_COMM_WORLD, &request[22]);
                MPI_Wait(&request[22], &status[22]);
            }
            if (rank == 7) {
                float u1_1_2;
                MPI_Isend(&u1, 1, MPI_FLOAT, 6, TAG, MPI_COMM_WORLD, &request[12]);
                
                MPI_Irecv(&u1_1_2, 1, MPI_FLOAT, 6, TAG, MPI_COMM_WORLD, &request[10]);
                MPI_Wait(&request[10], &status[10]);
                
                float u = G*u1_1_2;
                u2 = u1;
                u1 = u;
                u_matrix[matrix_row_index][matrix_column_index] = u;
                u1_matrix[matrix_row_index][matrix_column_index] = u1;
                u2_matrix[matrix_row_index][matrix_column_index] = u2;
            }
            if (rank == 8) {
                float u1_2_1;
                MPI_Isend(&u1, 1, MPI_FLOAT, 9, TAG, MPI_COMM_WORLD, &request[14]);
                
                MPI_Irecv(&u1_2_1, 1, MPI_FLOAT, 9, TAG, MPI_COMM_WORLD, &request[17]);
                MPI_Wait(&request[17], &status[17]);
                
                float u = G*u1_2_1;
                u2 = u1;
                u1 = u;
                u_matrix[matrix_row_index][matrix_column_index] = u;
                u1_matrix[matrix_row_index][matrix_column_index] = u1;
                u2_matrix[matrix_row_index][matrix_column_index] = u2;
                
                MPI_Isend(&u1, 1, MPI_FLOAT, 12, TAG, MPI_COMM_WORLD, &request[24]);
                MPI_Wait(&request[24], &status[24]);
            }
            if (rank == 11) {
                float u1_2_2;
                MPI_Isend(&u1, 1, MPI_FLOAT, 10, TAG, MPI_COMM_WORLD, &request[20]);
                
                MPI_Irecv(&u1_2_2, 1, MPI_FLOAT, 10, TAG, MPI_COMM_WORLD, &request[22]);
                MPI_Wait(&request[22], &status[22]);
                
                float u = G*u1_2_2;
                u2 = u1;
                u1 = u;
                u_matrix[matrix_row_index][matrix_column_index] = u;
                u1_matrix[matrix_row_index][matrix_column_index] = u1;
                u2_matrix[matrix_row_index][matrix_column_index] = u2;
            }
            if (rank == 13) {
                float u1_2_1;
                MPI_Isend(&u1, 1, MPI_FLOAT, 9, TAG, MPI_COMM_WORLD, &request[16]);
                
                MPI_Irecv(&u1_2_1, 1, MPI_FLOAT, 9, TAG, MPI_COMM_WORLD, &request[19]);
                MPI_Wait(&request[19], &status[19]);
                
                float u = G*u1_2_1;
                u2 = u1;
                u1 = u;
                u_matrix[matrix_row_index][matrix_column_index] = u;
                u1_matrix[matrix_row_index][matrix_column_index] = u1;
                u2_matrix[matrix_row_index][matrix_column_index] = u2;
            }
            if (rank == 14) {
                float u1_2_2;
                MPI_Isend(&u1, 1, MPI_FLOAT, 10, TAG, MPI_COMM_WORLD, &request[21]);
                
                MPI_Irecv(&u1_2_2, 1, MPI_FLOAT, 10, TAG, MPI_COMM_WORLD, &request[16]);
                MPI_Wait(&request[16], &status[16]);
                
                float u = G*u1_2_2;
                u2 = u1;
                u1 = u;
                u_matrix[matrix_row_index][matrix_column_index] = u;
                u1_matrix[matrix_row_index][matrix_column_index] = u1;
                u2_matrix[matrix_row_index][matrix_column_index] = u2;
                
                MPI_Isend(&u1, 1, MPI_FLOAT, 15, TAG, MPI_COMM_WORLD, &request[25]);
                MPI_Wait(&request[25], &status[25]);
            }
        }
        
        // For the corner nodes, the logic is even more conservative
        // A node just receives from a neighboring node(depends on which corner we're in)
        // And the node never sends any value
        if (rank == 0) {
            float u1_1_0;
            // receive value from process 4
            MPI_Irecv(&u1_1_0, 1, MPI_FLOAT, 4, TAG, MPI_COMM_WORLD, &request[22]);
            // block until receipt completes
            MPI_Wait(&request[22], &status[22]);
            
            // perform computation upon receipt and update necessary values
            float u = G*u1_1_0;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }
        // same applies for 3, 12, 15
        if (rank == 3) {
            float u1_0_2;
            MPI_Irecv(&u1_0_2, 1, MPI_FLOAT, 2, TAG, MPI_COMM_WORLD, &request[23]);
            MPI_Wait(&request[23], &status[23]);
            
            float u = G*u1_0_2;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }
        if (rank == 12) {
            float u1_2_0;
            MPI_Irecv(&u1_2_0, 1, MPI_FLOAT, 8, TAG, MPI_COMM_WORLD, &request[24]);
            MPI_Wait(&request[24], &status[24]);
            
            float u = G*u1_2_0;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;
        }
        if (rank == 15) {
            float u1_3_2;
            MPI_Irecv(&u1_3_2, 1, MPI_FLOAT, 14, TAG, MPI_COMM_WORLD, &request[25]);
            MPI_Wait(&request[25], &status[25]);
            
            float u = G*u1_3_2;
            u2 = u1;
            u1 = u;
            u_matrix[matrix_row_index][matrix_column_index] = u;
            u1_matrix[matrix_row_index][matrix_column_index] = u1;
            u2_matrix[matrix_row_index][matrix_column_index] = u2;        
        }
    }
    // cleans up all MPI states
    MPI_Finalize();  
    return 0;    
}
