//#include "clion_defines.cl"
//#include <CL/cl.h>
//#include <libclew/ocl_init.h>
//#include <stdio.h>
//
// as is bool matrix m * p
// bs is bool matrix p * n
// cs will be m * n
typedef unsigned char uchar;

#define TILE_SIDE 16 // as work group

__kernel void with_tiles(__global const uchar* as, __global const uchar* bs, __global uchar* cs,
                    const unsigned int m, const unsigned int p, const unsigned int  n)
{
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int row = get_group_id(0) * TILE_SIDE + local_row;
    int col = get_group_id(1) * TILE_SIDE + local_col;
    uchar res = false;

    __local uchar a_tile[TILE_SIDE][TILE_SIDE];
    __local uchar b_tile[TILE_SIDE][TILE_SIDE];

    unsigned int tail_cnt = (p + TILE_SIDE - 1) / TILE_SIDE;

//    printf("tail_cnt:  %d\n",tail_cnt);
    for (unsigned int t = 0; t < tail_cnt; ++t) {

//        printf("local_row:  %d, local_col: %d, row: %d, col: %d, as: %d, bs: %d\n",
//               local_row, local_col, row, col, p * row + (t * TILE_SIDE + local_col), n * (t * TILE_SIDE + local_row) + col);

        a_tile[local_row][local_col] = as[p * row + (t * TILE_SIDE + local_col)];
        b_tile[local_row][local_col] = bs[n * (t * TILE_SIDE + local_row) + col];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIDE; ++i) {
            res |= a_tile[local_row][i] & b_tile[i][local_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    cs[row * n + col] = res;
}