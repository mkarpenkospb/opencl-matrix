#include "clion_defines.cl"
#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <stdio.h>

// as is bool matrix m * p
// bs is bool matrix p * n
// cs will be m * n
typedef unsigned char uchar;

#define TILE_SIDE 16

__kernel void naive(__global const uchar* as, __global const uchar* bs, __global uchar* cs,
                     const unsigned int m, const unsigned int p, const unsigned int  n)
{
    int col = get_global_id(1);
    int row = get_global_id(0);
    uchar res = false;

    if (row >= m || col >= n) {
        return;
    }

    for (int i = 0; i < p; ++i) {
        res |= as[p * row + i] & bs[n * i + col];
    }

    cs[row * n + col] = res;
}