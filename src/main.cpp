#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <random>
#include "matrix_mult.hpp"

#include <vector>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>



void testMultiplication(cl::Device& device) {
    static const size_t mnp[3][3] = {{16, 16, 16}, {150, 200, 400}, {1000, 1000, 1000}};
    for (auto r : mnp) {

        size_t m = r[0];
        size_t p = r[1];
        size_t n = r[2];

        size_t tiled_m = (m + TILE_SIDE - 1) / TILE_SIDE * TILE_SIDE;
        size_t tiled_p = (p + TILE_SIDE - 1) / TILE_SIDE * TILE_SIDE;
        size_t tiled_n = (n + TILE_SIDE - 1) / TILE_SIDE * TILE_SIDE;

        size_t am_size = tiled_m * tiled_p;
        size_t bm_size = tiled_p * tiled_n;
        size_t cm_size = tiled_m * tiled_n;

        std::vector<uchar> am(am_size, 0);
        fillRandPad(am.data(), m, p, tiled_p);

        std::vector<uchar> bm(bm_size, 0);
        fillRandPad(bm.data(), p, n, tiled_n);

        std::vector<uchar> cm(cm_size, 0);

        matrix_mult(device, am, bm, cm, tiled_m, tiled_p, tiled_n, true);

        assert(checkMult(am.data(), bm.data(), cm.data(), tiled_m, tiled_p, tiled_n));
    }
}


void transitiveReductionSlow(cl::Device& device,
        const std::vector<uchar>& adj_matrix, std::vector<uchar>& result, size_t v) {
    std::vector<uchar> tmp(v * v);
    // задача -- возвести в степень не менее v, дальше не повлияет на результат
    while (v) {
        matrix_mult(device, result, result, tmp, v, v, v);
        result.swap(tmp);
        v >>= 1;
    }
}


void simpleTest() {
    cl::Device device = chooseDevice();
    size_t n = 5;
    size_t tiled_n = (n + TILE_SIDE - 1) / TILE_SIDE * TILE_SIDE;
    size_t am_size = tiled_n * tiled_n;
    std::vector<uchar> am = {
            1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    std::vector<uchar> perfect_result = {
            1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };


    std::vector<uchar> trRed(am.begin(), am.end());
    transitiveReduction(device, am, trRed, tiled_n);
    assert(std::equal(perfect_result.begin(), perfect_result.end(), trRed.begin()));
    std::cout <<  std::endl << "---simple test passed---" << std::endl;
}


int main() {
    std::cout << "MULTIPLICATION TEST" << std::endl;
    cl::Device device = chooseDevice();
    testMultiplication(device);
    std::cout <<  std::endl << "REDUCTION TEST" << std::endl;
    simpleTest();
    int size[] = {16, 1000};
    for (auto s: size) {
        size_t n = s;
        size_t tiled_n = (n + TILE_SIDE - 1) / TILE_SIDE * TILE_SIDE;
        size_t am_size = tiled_n * tiled_n;
        std::vector<uchar> am(am_size, 0);
        fillRandPad(am.data(), tiled_n, tiled_n, tiled_n, 0.9);
        std::vector<uchar> trRed(am.begin(), am.end());
        timer t;
        for (int i = 0; i < 20; ++i) {
            transitiveReduction(device, am, trRed, tiled_n);
            t.nextLap();
        }
        std::cout << "Average time for " << s << " x " <<  s << ": " << t.lapAvg() << std::endl;
        if (s == 16) {
            std::cout << "Adj: " << std::endl;
            showM(am.data(), tiled_n);
            std::cout << "Reduction: " << std::endl;
            showM(trRed.data(), tiled_n);
        }
    }
    return 0;
}






