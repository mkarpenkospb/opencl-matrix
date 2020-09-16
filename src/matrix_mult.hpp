#pragma once
#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <stdexcept>
#include <vector>
#include <CL/cl2.hpp>

using uchar = unsigned char;
typedef std::string STRING_CLASS;

#define TILE_SIDE 16

cl::Device chooseDevice();
void matrix_mult(cl::Device& device,
                 const std::vector<uchar>& am, const std::vector<uchar>& bm, std::vector<uchar>& cm,
                 unsigned int tiled_m, unsigned int tiled_p, unsigned int tiled_n, bool testMode = false);

void fillRandPad(uchar * matrix, size_t m, size_t n, size_t tiled_n, double percent = 0.5);
void fillIdentity(uchar * matrix, size_t n);
bool checkMult(const uchar* as, const uchar* bs, const uchar* cs, size_t m, size_t p, size_t n);
void multForTest(const uchar* as, const uchar* bs, uchar* cs, size_t m, size_t p, size_t n);
void transitiveReduction(cl::Device& device, const std::vector<uchar>& am, std::vector<uchar>& result, unsigned int n);
void showM(const uchar* matrix, size_t n);
void reportError(cl_int err, const std::string &filename, int line);
#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)