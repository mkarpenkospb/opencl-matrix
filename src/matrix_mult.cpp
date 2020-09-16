#include <fstream>
#include <libutils/timer.h>
#include <iostream>
#include <random>
#include "matrix_mult.hpp"


void showM(const uchar* matrix, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << (int) matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void fillRandPad(uchar * matrix, size_t m, size_t n, size_t tiled_n, double percent) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0, 1);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            matrix[i * tiled_n + j] = dist(mt) > percent;
        }
    }
}

void fillIdentity(uchar * matrix, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        matrix[i * n + i] = 1;
    }
}

void multForTest(const uchar* as, const uchar* bs, uchar* cs, size_t m, size_t p, size_t n) {
    uchar res;
    for (size_t i = 0; i < m; ++i ) {
        for (size_t j = 0; j < n; ++j) {
            res = 0;
            for (int k = 0; k < p; ++k) {
                res |= as[p * i + k] & bs[n * k + j];
            }
            cs[n * i + j] = res;
        }
    }
}


bool checkMult(const uchar* as, const uchar* bs, const uchar* cs, size_t m, size_t p, size_t n) {
    uchar res;
    for (size_t j = 0; j < m; ++j ) {
        for (size_t i = 0; i < n; ++i) {
            res = 0;
            for (int k = 0; k < p; ++k) {
                res |= as[p * j + k] & bs[n * k + i];
            }
            if ( cs[n * j + i] != res)
                return false;
        }
    }
    return true;
}


cl::Device chooseDevice() {
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");
    // -------------------------------------------- set up device ---------------------------------------------
    cl_uint platformsCount = 0;
    clGetPlatformIDs(0, nullptr, &platformsCount);
    if (platformsCount < 1)
        throw std::runtime_error("Can't run without a platform!");

    cl_platform_id platformId;
    clGetPlatformIDs(1, &platformId, nullptr);

    cl_device_id deviceId; //  CL_DEVICE_NOT_FOUND if no OpenCL devices that matched device_type were found
    clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 1, &deviceId, nullptr);

    return cl::Device(deviceId);
}

void reportError(cl_int err, const std::string &filename, int line)
{
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + std::to_string(err) + " encountered at " + filename + ":" + std::to_string(line);
    throw std::runtime_error(message);
}

void matrix_mult(cl::Device& device,
                 const std::vector<uchar>& am, const std::vector<uchar>& bm, std::vector<uchar>& cm,
                 unsigned int tiled_m, unsigned int tiled_p, unsigned int tiled_n, bool testMode) {
    cl_int err;
    size_t am_size = am.size();
    size_t bm_size = bm.size();
    size_t cm_size = cm.size();

    cl::Context context(device, nullptr, nullptr, nullptr, &err);
    OCL_SAFE_CALL(err);
    cl::CommandQueue queue(context, device, 0, &err);
    OCL_SAFE_CALL(err);

    cl::Buffer am_gpu(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, am_size * sizeof(uchar), (void *) am.data(), &err);
    OCL_SAFE_CALL(err);
    cl::Buffer bm_gpu(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bm_size * sizeof(uchar), (void *) bm.data());
    OCL_SAFE_CALL(err);
    cl::Buffer cm_gpu(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, cm_size * sizeof(uchar), cm.data());
    OCL_SAFE_CALL(err);
    STRING_CLASS kernel_sources;
    {
        std::ifstream file("../src/cl/with_tiles.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
    }

    cl::Program program(context, kernel_sources);
    OCL_SAFE_CALL(program.build());

    cl::Kernel kernel(program, "with_tiles", &err);
    OCL_SAFE_CALL(err);

    {
        unsigned int i = 0;
        kernel.setArg(i++, am_gpu);
        kernel.setArg(i++, bm_gpu);
        kernel.setArg(i++, cm_gpu);
        kernel.setArg(i++, sizeof(size_t), &tiled_m);
        kernel.setArg(i++, sizeof(size_t), &tiled_p);
        kernel.setArg(i++, sizeof(size_t), &tiled_n);
    }

    cl::NDRange workGroupSize = {TILE_SIDE, TILE_SIDE};
    cl::NDRange global_work_size = {
            (tiled_m + workGroupSize[0] - 1) / workGroupSize[0] * workGroupSize[0],
            (tiled_n + workGroupSize[1] - 1) / workGroupSize[1] * workGroupSize[1]
    };


    timer t;
    int end = testMode ? 20 : 1;
    for (unsigned int i = 0; i < end; ++i) {
        cl::Event eventEval;
        OCL_SAFE_CALL(queue.enqueueNDRangeKernel(kernel, 0, global_work_size, workGroupSize, nullptr, &eventEval));
        eventEval.wait();
    }
    if (testMode) {
        std::cout << "average time on m = " << tiled_m << ", p = " << tiled_p << ", n = " << tiled_n << " is " <<   t.lapAvg() << "+-"
                  << t.lapStd() << std::endl;
    }
    cl::Event eventGetResult;
    OCL_SAFE_CALL(queue.enqueueReadBuffer(cm_gpu, CL_FALSE, 0, cm_size * sizeof(uchar), cm.data(), nullptr, &eventGetResult));
    eventGetResult.wait();

}

void transitiveReduction(cl::Device& device, const std::vector<uchar>& am, std::vector<uchar>& result, unsigned int n) {

    cl_int err;
    size_t matrix_size = n * n;
    size_t log_steps = n;
    std::vector<uchar> tmp(result.begin(), result.end());

    cl::Context context(device, nullptr, nullptr, nullptr, &err);
    OCL_SAFE_CALL(err);
    cl::CommandQueue queue(context, device, 0, &err);
    OCL_SAFE_CALL(err);

    cl::Buffer am_gpu(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, matrix_size * sizeof(uchar), (void *) am.data(), &err);
    OCL_SAFE_CALL(err);
    cl::Buffer result_gpu(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, matrix_size * sizeof(uchar), (void *) result.data());
    OCL_SAFE_CALL(err);
    cl::Buffer tmp_gpu(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, matrix_size * sizeof(uchar), tmp.data());
    OCL_SAFE_CALL(err);

    STRING_CLASS kernel_sources;
    {
        std::ifstream file("../src/cl/with_tiles.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
    }

    cl::Program program(context, kernel_sources);
    OCL_SAFE_CALL(program.build());
    cl::Kernel kernel(program, "with_tiles", &err);
    OCL_SAFE_CALL(err);
    {
        unsigned int i = 0;
        kernel.setArg(i++, tmp_gpu);
        kernel.setArg(i++, tmp_gpu);
        kernel.setArg(i++, result_gpu);
        kernel.setArg(i++, sizeof(size_t), &n);
        kernel.setArg(i++, sizeof(size_t), &n);
        kernel.setArg(i++, sizeof(size_t), &n);
    }

    cl::NDRange workGroupSize = {TILE_SIDE, TILE_SIDE};
    cl::NDRange global_work_size = {
            (n + workGroupSize[0] - 1) / workGroupSize[0] * workGroupSize[0],
            (n + workGroupSize[1] - 1) / workGroupSize[1] * workGroupSize[1]
    };

    bool coutner = true;
    while (log_steps) {
        cl::Event eventEval;
        OCL_SAFE_CALL(queue.enqueueNDRangeKernel(kernel, 0, global_work_size, workGroupSize, nullptr, &eventEval));
        eventEval.wait();
        if (coutner) {
            kernel.setArg(0, result_gpu);
            kernel.setArg(1, result_gpu);
            kernel.setArg(2, tmp_gpu);
            coutner = false;
        } else {
            kernel.setArg(0, tmp_gpu);
            kernel.setArg(1, tmp_gpu);
            kernel.setArg(2, result_gpu);
            coutner = true;
        }

        log_steps >>= 1;
    }

    cl::Event eventGetResult;
    OCL_SAFE_CALL(queue.enqueueReadBuffer(coutner ? result_gpu : tmp_gpu, CL_FALSE, 0,  matrix_size * sizeof(uchar), result.data(), nullptr, &eventGetResult));
    eventGetResult.wait();
}
