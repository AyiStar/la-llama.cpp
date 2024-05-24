#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <locale.h>
#include <assert.h>
#include <math.h>
#include <cstring>
#include <cstdio>
#include <cinttypes>
#include <unordered_map>
#include <queue>
#include <string.h>
#include <cassert>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

static float tensor_sum_elements(const ggml_tensor * tensor) {
    double sum = 0;
    if (tensor->type == GGML_TYPE_F32) {
        for (int j = 0; j < tensor->ne[1]; j++) {
            for (int k = 0; k < tensor->ne[0]; k++) {
                sum += ((float *) tensor->data)[j*tensor->ne[0] + k];
            }
        }
    }
    return sum;
}

static void tensor_dump(const ggml_tensor * tensor, const char * name) {
    printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi) - ", name,
        tensor->type, ggml_type_name(tensor->type),
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->nb[0], tensor->nb[1], tensor->nb[2]);
    float sum = tensor_sum_elements(tensor);
    printf("Sum of tensor %s is %6.2f\n", name, sum);
}

#define TENSOR_DUMP(tensor) tensor_dump(tensor, #tensor)

struct benchmark_params_struct {
    int32_t n_threads     = 1;
    int32_t n_iterations  = 10;
};

static void print_usage(int /*argc*/, char ** argv, struct benchmark_params_struct params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -i N, --iter N     number of iterations to use during computation (default: %d)\n", params.n_iterations);
    fprintf(stderr, "\n");
}

static void do_test(
    int test_no,
    int sizex,
    int sizey,
    int sizez,
    ggml_type type,
    struct ggml_tensor *m11,
    struct ggml_tensor *m12,
    struct ggml_tensor *m2,
    struct ggml_context *ctx,
    struct benchmark_params_struct benchmark_params,
    std::vector<uint8_t>& work_buffer,
    const float correct
);

static void do_benchmark(
    int sizex,
    int sizey,
    int sizez,
    struct ggml_cgraph * g1,
    struct ggml_cgraph * g2,
    struct benchmark_params_struct benchmark_params,
    std::vector<uint8_t>& work_buffer,
    const float correct
);

int main(int argc, char ** argv)  {
    struct benchmark_params_struct benchmark_params;

    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-i" || arg == "--iter") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_iterations = std::stoi(argv[i]);
        }  else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, benchmark_params);
            exit(0);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv, benchmark_params);
        exit(1);
    }

    print_build_info();
    // print system information
    {
        LOG_TEE("\n");
        LOG_TEE("%s\n", llama_print_system_info());
    }
    printf("Starting Test\n");

    // create the ggml context
    struct ggml_context * ctx;
    //const int sizex = 4096;
    //const int sizey = 11008;

#undef VERBOSE_DEBUGGING
#ifndef VERBOSE_DEBUGGING
    // const int sizey = 4096;
    // const int sizex = 11008;
    // const int sizez = 128;

    const int sizey = 1;
    const int sizex = 128;
    const int sizez = 1;
#else
    /* Working - let's increase size */
    const int sizey = 1;
    const int sizex = (8*32);
    const int sizez = 1;

    /*const int sizey = 1;
    const int sizex = 3*(8*32);
    const int sizez = 1;*/
#endif

    //printf("Memsize required = %i\n", sizex*sizex);

    // TODO: perform the bench for all types or for a user specified type
    const ggml_type qtype = GGML_TYPE_Q4_1;

    size_t ctx_size = 0;
    ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizey);
    ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizey);
    ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizez);
    ctx_size += ggml_row_size(qtype,         sizex*sizey);
    ctx_size += ggml_row_size(qtype,         sizex*sizey);
    ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizey); // BLAS
    ctx_size += ggml_row_size(GGML_TYPE_F32, sizex*sizey); // BLAS
    ctx_size += 1024*1024*16;

    printf("Allocating Memory of size %zi bytes, %zi MB\n",ctx_size, (ctx_size/1024/1024));

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /* no_alloc   =*/ 0
    };

    ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return 1;
    }


    printf("Creating new tensors\n");
    // printf("Creating new tensor m1\n");
    struct ggml_tensor * m11 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizey);
    ggml_set_f32(m11, 1.0f);

    // printf("Creating new tensor m1\n");
    struct ggml_tensor * m12 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizey);
    ggml_set_f32(m12, 1.5f);

    // printf("Creating new tensor m2\n");
    struct ggml_tensor * m2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizez);
    ggml_set_f32(m2, 2.0f);

    const double correct_sum_m11xm2 = (sizex * (1.0f * 2.0f)) * (sizey * sizez);
    printf("Theoretical sum of m11xm2 = %6.2f\n", correct_sum_m11xm2);

    printf("\n------ Test 0 - Matrix Mult via F32 code\n");
    // printf("Creating new tensor m11xm2\n");
    struct ggml_tensor * m11xm2 = ggml_mul_mat(ctx, m11, m2);

    // printf("Creating compute graph\n");
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, m11xm2);

    printf("n_threads=%i\n", benchmark_params.n_threads);

    TENSOR_DUMP(m11);
    TENSOR_DUMP(m2);

    std::vector<uint8_t> work_buffer;

    ggml_graph_compute_helper(work_buffer, gf, benchmark_params.n_threads);

    TENSOR_DUMP(gf->nodes[0]);

    [[maybe_unused]] float sum_of_F32_reference = tensor_sum_elements(gf->nodes[0]);
    assert (std::abs(sum_of_F32_reference - correct_sum_m11xm2) < 1e-6);

    // test on F32
    // do_test(1, sizex, sizey, sizez, GGML_TYPE_F32, m11, m12, m2, ctx, benchmark_params, work_buffer, correct_sum_m11xm2);

    // test on Q4_1
    do_test(2, sizex, sizey, sizez, qtype, m11, m12, m2, ctx, benchmark_params, work_buffer, correct_sum_m11xm2);
}


void do_test(
    int test_no,
    int sizex,
    int sizey,
    int sizez,
    ggml_type type,
    struct ggml_tensor *m11,
    struct ggml_tensor *m12,
    struct ggml_tensor *m2,
    struct ggml_context *ctx,
    struct benchmark_params_struct benchmark_params,
    std::vector<uint8_t>& work_buffer,
    const float correct
) {

    printf("\n------ Test %d - Matrix Mult via %s code\n", test_no, ggml_type_name(type));

    int32_t nelements = sizex*sizey;

    // Set up a the benchmark matrices

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        struct ggml_tensor * q11 = ggml_new_tensor_2d(ctx, type, sizex, sizey);
        ggml_quantize_chunk(type, (const float *) m11->data, q11->data, 0, nelements/m11->ne[0], m11->ne[0], nullptr);
        struct ggml_tensor * q12 = ggml_new_tensor_2d(ctx, type, sizex, sizey);
        ggml_quantize_chunk(type, (const float *) m12->data, q12->data, 0, nelements/m12->ne[0], m12->ne[0], nullptr);
        m11 = q11;
        m12 = q12;
    }

    // Set up a the compute graph
    struct ggml_tensor * m11xm2 = ggml_mul_mat(ctx, m11, m2);

    struct ggml_cgraph * g1 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g1, m11xm2);

    // Set up a second graph computation to make sure we override the CPU cache lines
    struct ggml_tensor * m12xm2 = ggml_mul_mat(ctx, m12, m2);

    struct ggml_cgraph * g2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g2, m12xm2);
    printf("n_threads=%i\n", benchmark_params.n_threads);

    const int dimx = sizex;
    const int dimy = sizey;
    const int dimz = sizez;
    long long int flops_per_dot_product = dimy + dimy;
    long long int flops_per_matrix = flops_per_dot_product * dimx * dimz; ;
    printf("Matrix Multiplication of (%i,%i,%i) x (%i,%i,%i) - about %6.2f gFLOPS\n\n", sizex, sizey, 1, sizex, sizez, 1, 1.0f*flops_per_matrix / 1000 / 1000 / 1000);

    do_benchmark(sizex, sizey, sizez, g1, g2, benchmark_params, work_buffer, correct);
}


void do_benchmark(
    int sizex,
    int sizey,
    int sizez,
    struct ggml_cgraph * g1,
    struct ggml_cgraph * g2,
    struct benchmark_params_struct benchmark_params,
    std::vector<uint8_t>& work_buffer,
    const float correct
) {
    printf("n_threads=%i\n", benchmark_params.n_threads);

    const int dimx = sizex;
    const int dimy = sizey;
    const int dimz = sizez;
    long long int flops_per_dot_product = dimy + dimy;
    long long int flops_per_matrix = flops_per_dot_product * dimx * dimz; ;
    printf("Matrix Multiplication of (%i,%i,%i) x (%i,%i,%i) - about %6.2f gFLOPS\n\n", sizex, sizey, 1, sizex, sizez, 1, 1.0f*flops_per_matrix / 1000 / 1000 / 1000);

    printf("Iteration;NThreads; SizeX; SizeY; SizeZ; Required_FLOPS; Elapsed_u_Seconds; gigaFLOPS\n");
    printf("=====================================================================================\n");

    double  gflops_sum = 0;
    for (int i=0;i<benchmark_params.n_iterations ;i++) {

        long long int start = ggml_time_us();
        //printf("Running ggml_graph_compute\n");
        ggml_graph_compute_helper(work_buffer, g1, benchmark_params.n_threads);

        long long int stop = ggml_time_us();
        long long int usec = stop-start;
        double gflops = (double)(flops_per_matrix)/usec/1000.0;
        gflops_sum += gflops;
        printf("%9i;%8i;%6i;%6i;%6i;%15lli;%18lli;%10.2f\n",
            i,
            benchmark_params.n_threads,
            sizex, sizey, sizez, flops_per_matrix,
            usec,gflops);

#ifdef VERBOSE_DEBUGGING
        TENSOR_DUMP("res",g1.nodes[0])
#endif

        // Check that the matrix multiplication result is in the right ballpark
        // We cannot use the exact value from the F32 multiplication because the quantizuation will be slightly different
        float sum_of_result = tensor_sum_elements(g1->nodes[0]);
        float delta = std::abs(sum_of_result - correct);
        float allowed_delta = (correct) / 1000 / 1000; //  Let's accept an epsilon of 10^-6

        if (delta > allowed_delta)  {
            printf("\nABORT - ERROR in Matrix Multiplication result - expected %6.2f, got %6.2f (delta %6.2f > allowed_delta %6.2f)\n",
                correct,
                sum_of_result,
                delta,
                allowed_delta
            );
            exit(0);
        }

        // Running a different graph computation to make sure we override the CPU cache lines
        ggml_graph_compute_helper(work_buffer, g2, benchmark_params.n_threads);
    }
    printf("\n");
    printf("Average%78.2f\n",gflops_sum/((double)benchmark_params.n_iterations));
    printf("=====================================================================================\n");
}