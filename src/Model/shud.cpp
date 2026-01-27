#include <stdio.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <chrono>
#include <iostream>
//#include "f_element.hpp"
//#include "f_River.hpp"
#include "f.hpp"
#include "IO.hpp"
#include "ModelConfigure.hpp"
#include "print.hpp"
#include "Macros.hpp"
#include "functions.hpp"
//#include "is_sm_et.hpp"
#include "cvode_config.hpp"
#include "Model_Data.hpp"
#ifdef _CUDA_ON
#include "DeviceContext.hpp"
#endif
#include "TimeSeriesData.hpp"
#include "FloodAlert.hpp"
#include "CommandIn.hpp"

double *uYsf;
double *uYus;
double *uYgw;
double *uYriv;
double *uYlake;
double *globalY;
double timeNow;
int dummy_mode = 0;
int global_fflush_mode = 0;
int global_implicit_mode = 1;
int global_verbose_mode = 1;
/* Default backend based on compile-time configuration.
 * - shud_cuda (_CUDA_ON): defaults to CUDA
 * - shud_omp (_OPENMP_ON, no _CUDA_ON): defaults to OMP
 * - shud (neither): defaults to CPU
 * User can override via --backend flag.
 */
#if defined(_CUDA_ON)
int global_backend = BACKEND_CUDA;
#elif defined(_OPENMP_ON)
int global_backend = BACKEND_OMP;
#else
int global_backend = BACKEND_CPU;
#endif
int global_precond_enabled = 1; /* Whether to use CVODE preconditioning (CUDA backend only). */
int global_backend_cli_set = 0; /* Whether --backend is explicitly set by CLI. */
int lakeon = 0; /* Whether lake module ON(1), OFF(0) */
int CLAMP_POLICY = 1; /* Whether to clamp state to non-negative values */
int CLAMP_POLICY_CLI_SET = 0; /* Whether CLAMP_POLICY is overridden by CLI (-C) */
using namespace std;

static bool iequalsN(const char *a, size_t n, const char *b)
{
    if (a == NULL || b == NULL) {
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        if (b[i] == '\0') {
            return false;
        }
        const unsigned char ca = static_cast<unsigned char>(a[i]);
        const unsigned char cb = static_cast<unsigned char>(b[i]);
        if (tolower(ca) != tolower(cb)) {
            return false;
        }
    }
    return b[n] == '\0';
}

static bool parseBoolEnvValue(const char *value, int *out)
{
    if (out == NULL || value == NULL) {
        return false;
    }

    const char *start = value;
    while (*start != '\0' && isspace(static_cast<unsigned char>(*start))) {
        start++;
    }
    if (*start == '\0') {
        return false;
    }
    const char *end = start;
    while (*end != '\0') {
        end++;
    }
    while (end > start && isspace(static_cast<unsigned char>(end[-1]))) {
        end--;
    }
    const size_t len = static_cast<size_t>(end - start);
    if (len == 0) {
        return false;
    }

    errno = 0;
    char *num_end = NULL;
    const long v = strtol(start, &num_end, 10);
    if (errno == 0 && num_end != NULL && num_end != start) {
        while (*num_end != '\0' && isspace(static_cast<unsigned char>(*num_end))) {
            num_end++;
        }
        if (*num_end == '\0') {
            *out = (v == 0) ? 0 : 1;
            return true;
        }
    }

    if (iequalsN(start, len, "true") || iequalsN(start, len, "on") || iequalsN(start, len, "yes")) {
        *out = 1;
        return true;
    }
    if (iequalsN(start, len, "false") || iequalsN(start, len, "off") || iequalsN(start, len, "no")) {
        *out = 0;
        return true;
    }

    return false;
}

static int parsePositiveIntEnv(const char *name, int fallback)
{
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    errno = 0;
    char *end = NULL;
    const long v = strtol(value, &end, 10);
    if (errno != 0 || end == value) {
        return fallback;
    }
    while (*end != '\0' && isspace(static_cast<unsigned char>(*end))) {
        end++;
    }
    if (*end != '\0') {
        return fallback;
    }
    if (v <= 0 || v > INT_MAX) {
        return fallback;
    }
    return static_cast<int>(v);
}

double SHUD(FileIn *fin, FileOut *fout){
    double ret = 0.;
    Model_Data  *MD;        /* Model Data                */
    N_Vector    udata;
    N_Vector    du;

#ifdef _CUDA_ON
    SUNCudaExecPolicy *nvec_stream_exec_policy = nullptr;
    SUNCudaExecPolicy *nvec_reduce_exec_policy = nullptr;
#endif
    
    SUNContext sunctx;
    ret = SUNContext_Create(NULL, &sunctx);
    check_flag(&ret, "SUNContext_Create", 1);
    
    void    *mem = NULL;
    SUNLinearSolver LS = NULL;
    int     flag;            /* flag to test return value */
    double  t, tnext;    /* stress period & step size */
    int NY = 0;
    int ierr = 0;
    /* allocate memory for model data structure */
    MD = new Model_Data(fin, fout);
    MD->loadinput();
    MD->initialize();
    MD->CheckInputData();
    fout->updateFilePath();
    NY = MD->NumY;

    {
        /* Backend auto-selection for small problems (avoid slow CUDA path on tiny NY).
         *
         * Default behavior:
         * - If this binary defaults to CUDA (global_backend==BACKEND_CUDA) and the user did not
         *   explicitly pass --backend, auto-selection is enabled unless SHUD_BACKEND_AUTO disables it.
         *
         * Controls:
         * - Env: SHUD_BACKEND_AUTO=0/1 (also accepts true/false/on/off/yes/no)
         * - Env: NY_GPU_MIN=<positive int> (default 100000)
         * - Override: --backend cuda (forces CUDA even when auto-selection would choose CPU/OMP)
         */
        bool auto_enabled = false;
        const char *auto_env = getenv("SHUD_BACKEND_AUTO");
        int auto_env_value = 0;
        if (auto_env != NULL && parseBoolEnvValue(auto_env, &auto_env_value)) {
            auto_enabled = (auto_env_value != 0);
        } else {
            auto_enabled = (global_backend == BACKEND_CUDA && !global_backend_cli_set);
        }

        if (auto_enabled && global_backend == BACKEND_CUDA && !global_backend_cli_set) {
            const int ny_gpu_min = parsePositiveIntEnv("NY_GPU_MIN", 100000);
            if (NY > 0 && NY < ny_gpu_min) {
                int fallback_backend = BACKEND_CPU;
#ifdef _OPENMP_ON
                fallback_backend = BACKEND_OMP;
#endif
                global_backend = fallback_backend;
                {
                    char msg[MAXLEN];
                    snprintf(msg,
                             sizeof(msg),
                             "Backend auto-select: NY=%d < NY_GPU_MIN=%d, switching to %s (override with --backend cuda)\n",
                             NY,
                             ny_gpu_min,
                             (fallback_backend == BACKEND_OMP) ? "omp" : "cpu");
                    screeninfo(msg);
                }
            } else {
                {
                    char msg[MAXLEN];
                    snprintf(msg,
                             sizeof(msg),
                             "Backend auto-select: NY=%d >= NY_GPU_MIN=%d, using cuda\n",
                             NY,
                             ny_gpu_min);
                    screeninfo(msg);
                }
            }
        } else if (auto_enabled && global_backend_cli_set) {
            screeninfo("Backend auto-select: enabled but skipped (explicit --backend provided)\n");
        }
    }

    const int nthreads = max(MD->CS.num_threads, 1);
    switch (global_backend) {
        case BACKEND_CPU:
#ifdef _OPENMP_ON
            /* Keep OpenMP-parallel code paths deterministic-ish when using the CPU backend. */
            MD->CS.num_threads = 1;
            omp_set_num_threads(1);
#endif
            screeninfo("\nBackend: cpu (NVECTOR_SERIAL)\n");
            udata = N_VNew_Serial(NY, sunctx);
            du = N_VNew_Serial(NY, sunctx);
            break;
        case BACKEND_OMP:
#ifdef _OPENMP_ON
        {
            /*
             * Reproducibility note:
             * SUNDIALS NVECTOR_OPENMP parallel reductions (dot products, norms, etc.)
             * can introduce small run-to-run floating-point differences due to
             * non-associative summation order. These differences can cascade through
             * the adaptive solver and make outputs non-bitwise-reproducible.
             *
             * The hydrologic flux kernels below are parallelized explicitly with
             * OpenMP (num_threads=CS.num_threads). To prioritize deterministic
             * results, keep NVECTOR_OPENMP math reductions single-threaded by
             * default. Override via env var SHUD_NVEC_THREADS.
             */
            const int nvec_threads = parsePositiveIntEnv("SHUD_NVEC_THREADS", 1);
            omp_set_num_threads(nthreads);
            {
                char msg[MAXLEN];
                snprintf(msg,
                         sizeof(msg),
                         "\nBackend: omp (NVECTOR_OPENMP). Threads = %d (RHS), %d (NVECTOR)\n",
                         nthreads,
                         nvec_threads);
                screeninfo(msg);
            }
            udata = N_VNew_OpenMP(NY, nvec_threads, sunctx);
            du = N_VNew_OpenMP(NY, nvec_threads, sunctx);
            break;
        }
#else
            fprintf(stderr, "\nERROR: --backend omp requested, but this build does not enable OpenMP.\n\n");
            myexit(-1);
#endif
        case BACKEND_CUDA:
#ifdef _CUDA_ON
            screeninfo("\nBackend: cuda (NVECTOR_CUDA)\n");
            {
                static bool cuda_experimental_warned = false;
                if (!cuda_experimental_warned) {
                    screeninfo("NOTE: CUDA backend is experimental. Results may differ slightly from CPU.\n");
                    cuda_experimental_warned = true;
                }
            }
            {
                char msg[MAXLEN];
                snprintf(msg,
                         sizeof(msg),
                         "CUDA preconditioner: %s (env SHUD_CUDA_PRECOND=0 or --no-precond to disable)\n",
                         global_precond_enabled ? "ON" : "OFF");
                screeninfo(msg);
            }
            udata = N_VNew_Cuda(NY, sunctx);
            du = N_VNew_Cuda(NY, sunctx);
            check_flag((void *)udata, "N_VNew_Cuda", 0);
            check_flag((void *)du, "N_VNew_Cuda", 0);

            /* Access device pointer to validate NVECTOR_CUDA data layout. */
            if (N_VGetDeviceArrayPointer_Cuda(udata) == NULL) {
                fprintf(stderr, "\nSUNDIALS_ERROR: N_VGetDeviceArrayPointer_Cuda() returned NULL\n\n");
                myexit(ERRCVODE);
            }
            if (N_VIsManagedMemory_Cuda(udata)) {
                screeninfo("WARNING: NVECTOR_CUDA is using managed memory (UVM).\n");
            } else {
                screeninfo("NVECTOR_CUDA memory: unmanaged device memory\n");
            }

            gpuInit(MD);
            if (MD->cuda_stream != nullptr) {
                /* SUNDIALS 6.0 NVECTOR_CUDA uses kernel exec policies (stream is embedded).
                 * Keep policies alive for the lifetime of the NVECTORs. */
                nvec_stream_exec_policy = new SUNCudaThreadDirectExecPolicy(256, MD->cuda_stream);
                nvec_reduce_exec_policy = new SUNCudaBlockReduceExecPolicy(256, 0, MD->cuda_stream);

                int policy_flag = N_VSetKernelExecPolicy_Cuda(udata, nvec_stream_exec_policy, nvec_reduce_exec_policy);
                check_flag(&policy_flag, "N_VSetKernelExecPolicy_Cuda", 1);
                policy_flag = N_VSetKernelExecPolicy_Cuda(du, nvec_stream_exec_policy, nvec_reduce_exec_policy);
                check_flag(&policy_flag, "N_VSetKernelExecPolicy_Cuda", 1);
            }
            break;
#else
            fprintf(stderr, "\nERROR: --backend cuda requested, but this build does not enable CUDA (NVECTOR_CUDA).\n\n");
            myexit(-1);
#endif
        default:
            fprintf(stderr, "\nERROR: unknown backend=%d (expect cpu|omp|cuda).\n\n", global_backend);
            myexit(-1);
    }

    screeninfo("\nGlobal Implicit Mode: ON\n");
    MD->LoadIC();
    MD->SetIC2Y(udata);
#ifdef _CUDA_ON
    if (N_VGetVectorID(udata) == SUNDIALS_NVEC_CUDA) {
        /* Initial conditions were set on host; sync to device for CVODE. */
        N_VCopyToDevice_Cuda(udata);
    }
#endif
    MD->initialize_output();
    MD->PrintInit(fout->Init_bak, 0);
    MD->InitFloodAlert(fout->floodout);
    SetCVODE(mem, f, MD, udata, LS, sunctx);
    /* set start time */
    t = MD->CS.StartTime;
    tnext = t;
    //CheckInput(MD, &CS);
    /* start solver in loops */
//    getSecond();
    MD->modelSummary(0);
    MD->debugData(fout->outpath);
    MD->gc.write(fout->Calib_bak);

    const auto bench_wall_start = std::chrono::steady_clock::now();
    double bench_forcing_s = 0.0;
    double bench_cvode_s = 0.0;
    double bench_io_s = 0.0;
//    f(t, udata, du, MD); /* Initialized the status */
    const bool etSubstepEnabled =
        (MD->CS.ETStep > ZERO && MD->CS.ETStep + ZERO < MD->CS.SolverStep);
    for (int i = 0; i < MD->CS.NumSteps && !ierr; i++) {
        printDY(MD->file_debug);
#ifdef DEBUG
        printDY(MD->file_debug);
#endif
        flag = MD->ScreenPrint(t, i);
        {
            const auto io_start = std::chrono::steady_clock::now();
            MD->PrintInit(fout->Init_update, t);
            bench_io_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - io_start).count();
        }
        /* inner loops to next output points with ET step size control */
        tnext += MD->CS.SolverStep;
        while (t + ZERO < tnext) {
            double tout = tnext;
            if (etSubstepEnabled) {
                tout = min(t + MD->CS.ETStep, tnext);
            }

            const auto forcing_start = std::chrono::steady_clock::now();
            MD->updateAllTimeSeries(t);
            MD->updateBC(t);
            MD->updateforcing(t);
            /* calculate Interception Storage */
            MD->ET(t, tout);
#ifdef _CUDA_ON
            if (N_VGetVectorID(udata) == SUNDIALS_NVEC_CUDA) {
                MD->gpuUpdateForcing();
            }
#endif
            bench_forcing_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - forcing_start).count();
            if (dummy_mode) {
                t = tout; /* dummy mode only. */
            } else {
                if (etSubstepEnabled) {
                    flag = CVodeSetStopTime(mem, tout);
                    check_flag(&flag, "CVodeSetStopTime", 1);
                }
                const auto cvode_start = std::chrono::steady_clock::now();
                flag = CVode(mem, tout, udata, &t, CV_NORMAL);
                bench_cvode_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - cvode_start).count();
                check_flag(&flag, "CVode", 1);
            }
        }
        //            CVODEstatus(mem, udata, t);
#ifdef _CUDA_ON
        const auto io_sync_start = std::chrono::steady_clock::now();
        MD->gpuSyncStateFromDevice(udata);
        MD->gpuSyncDiagnosticsFromDevice(udata);
        bench_io_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - io_sync_start).count();
#endif
        {
            const auto io_start = std::chrono::steady_clock::now();
            MD->summary(udata);
            MD->CS.ExportResults(t);
            MD->flood->FloodWarning(t);
            bench_io_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - io_start).count();
        }
    }
    MD->ScreenPrint(t, MD->CS.NumSteps);
    {
        const auto io_start = std::chrono::steady_clock::now();
        MD->PrintInit(fout->Init_update, t);
        bench_io_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - io_start).count();
    }

    {
        long int nfe = -1;
        long int nli = -1;
        long int nni = -1;
        long int netf = -1;
        long int npe = -1;
        long int nps = -1;

        int stats_flag = CVodeGetNumRhsEvals(mem, &nfe);
        if (stats_flag != 0) {
            fprintf(stderr, "WARNING: CVodeGetNumRhsEvals failed with flag=%d (continuing)\n", stats_flag);
        }

        stats_flag = CVodeGetNumLinIters(mem, &nli);
        if (stats_flag != 0) {
            fprintf(stderr, "WARNING: CVodeGetNumLinIters failed with flag=%d (continuing)\n", stats_flag);
        }

        stats_flag = CVodeGetNumNonlinSolvIters(mem, &nni);
        if (stats_flag != 0) {
            fprintf(stderr, "WARNING: CVodeGetNumNonlinSolvIters failed with flag=%d (continuing)\n", stats_flag);
        }

        stats_flag = CVodeGetNumErrTestFails(mem, &netf);
        if (stats_flag != 0) {
            fprintf(stderr, "WARNING: CVodeGetNumErrTestFails failed with flag=%d (continuing)\n", stats_flag);
        }

        stats_flag = CVodeGetNumPrecEvals(mem, &npe);
        if (stats_flag != 0) {
            fprintf(stderr, "WARNING: CVodeGetNumPrecEvals failed with flag=%d (continuing)\n", stats_flag);
        }

        stats_flag = CVodeGetNumPrecSolves(mem, &nps);
        if (stats_flag != 0) {
            fprintf(stderr, "WARNING: CVodeGetNumPrecSolves failed with flag=%d (continuing)\n", stats_flag);
        }

        printf("\nCVODE_STATS nfe=%ld nli=%ld nni=%ld netf=%ld npe=%ld nps=%ld\n\n", nfe, nli, nni, netf, npe, nps);
    }

    const double bench_wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - bench_wall_start).count();
    printf("\nBENCH_STATS wall_s=%.6f cvode_s=%.6f io_s=%.6f forcing_s=%.6f\n\n",
           bench_wall_s,
           bench_cvode_s,
           bench_io_s,
           bench_forcing_s);

    MD->modelSummary(1);
    /* Free memory */
    N_VDestroy(udata);
    N_VDestroy(du);
#ifdef _CUDA_ON
    delete nvec_stream_exec_policy;
    delete nvec_reduce_exec_policy;
#endif
    /* Free integrator memory */
    CVodeFree(&mem);
    SUNLinSolFree(LS);

    SUNContext_Free(&sunctx);
    delete MD;
    return ret;
}


double SHUD_uncouple(FileIn *fin, FileOut *fout){
    double ret = 0.;
    Model_Data  *MD;        /* Model Data                */
    N_Vector    u1, u2, u3, u4, u5;
    N_Vector    du1, du2, du3, du4, du5;
    SUNContext sunctx1, sunctx2, sunctx3, sunctx4, sunctx5;
    ret = SUNContext_Create(NULL, &sunctx1);check_flag(&ret, "SUNContext_Create", 1);
    ret = SUNContext_Create(NULL, &sunctx2);check_flag(&ret, "SUNContext_Create", 1);
    ret = SUNContext_Create(NULL, &sunctx3);check_flag(&ret, "SUNContext_Create", 1);
    ret = SUNContext_Create(NULL, &sunctx4);check_flag(&ret, "SUNContext_Create", 1);
    ret = SUNContext_Create(NULL, &sunctx5);check_flag(&ret, "SUNContext_Create", 1);
    
    void    *mem1 = NULL, *mem2 = NULL, *mem3 = NULL, *mem4 = NULL, *mem5 = NULL;
    SUNLinearSolver LS1 = NULL, LS2 = NULL, LS3 = NULL, LS4 = NULL, LS5 = NULL;
    int     flag;            /* flag to test return value */
    double  t = 0, dt = 0, tout = 0;    /* stress period & step size */
    int NY = 0;
    int N1, N2, N3, N4, N5;
    int ierr = 0;
    /* allocate memory for model data structure */
    MD = new Model_Data(fin, fout);
    MD->loadinput();
    MD->initialize();
    MD->CheckInputData();
    fout->updateFilePath();
    NY = MD->NumY;
    N1 = MD->NumEle;
    N2 = MD->NumEle;
    N3 = MD->NumEle;
    N4 = MD->NumRiv;
    N5 = MD->NumLake;

    const int nthreads = max(MD->CS.num_threads, 1);
    const int nvec_threads = parsePositiveIntEnv("SHUD_NVEC_THREADS", 1);

    if (global_backend == BACKEND_CUDA) {
        fprintf(stderr,
                "\nERROR: --backend cuda is not supported in uncoupled mode (-g). "
                "Please run coupled mode or use --backend cpu/omp.\n\n");
        myexit(-1);
    }

    if (global_backend == BACKEND_OMP) {
#ifdef _OPENMP_ON
        omp_set_num_threads(nthreads);
        {
            char msg[MAXLEN];
            snprintf(msg,
                     sizeof(msg),
                     "\nBackend: omp (NVECTOR_OPENMP). Threads = %d (RHS), %d (NVECTOR)\n",
                     nthreads,
                     nvec_threads);
            screeninfo(msg);
        }
#else
        fprintf(stderr, "\nERROR: --backend omp requested, but this build does not enable OpenMP.\n\n");
        myexit(-1);
#endif
    } else {
#ifdef _OPENMP_ON
        MD->CS.num_threads = 1;
        omp_set_num_threads(1);
#endif
        screeninfo("\nBackend: cpu (NVECTOR_SERIAL)\n");
    }

    screeninfo("\nGlobal Implicit Mode: OFF\n");

    if (global_backend == BACKEND_OMP) {
#ifdef _OPENMP_ON
        u1 = N_VNew_OpenMP(N1, nvec_threads, sunctx1);
        u2 = N_VNew_OpenMP(N2, nvec_threads, sunctx2);
        u3 = N_VNew_OpenMP(N3, nvec_threads, sunctx3);
        u4 = N_VNew_OpenMP(N4, nvec_threads, sunctx4);
        u5 = N_VNew_OpenMP(N5, nvec_threads, sunctx5);

        du1 = N_VNew_OpenMP(N1, nvec_threads, sunctx1);
        du2 = N_VNew_OpenMP(N2, nvec_threads, sunctx2);
        du3 = N_VNew_OpenMP(N3, nvec_threads, sunctx3);
        du4 = N_VNew_OpenMP(N4, nvec_threads, sunctx4);
        du5 = N_VNew_OpenMP(N5, nvec_threads, sunctx5);
#endif
    } else {
        u1 = N_VNew_Serial(N1, sunctx1);
        u2 = N_VNew_Serial(N2, sunctx2);
        u3 = N_VNew_Serial(N3, sunctx3);
        u4 = N_VNew_Serial(N4, sunctx4);
        u5 = N_VNew_Serial(N5, sunctx5);

        du1 = N_VNew_Serial(N1, sunctx1);
        du2 = N_VNew_Serial(N2, sunctx2);
        du3 = N_VNew_Serial(N3, sunctx3);
        du4 = N_VNew_Serial(N4, sunctx4);
        du5 = N_VNew_Serial(N5, sunctx5);
    }

    MD->LoadIC();
    MD->SetIC2Y(u1, u2, u3, u4, u5);
    MD->initialize_output();
    MD->PrintInit(fout->Init_bak, 0);
    MD->InitFloodAlert(fout->floodout);
    
    SetCVODE(mem1, f_surf,  MD, u1, LS1, sunctx1);
    SetCVODE(mem2, f_unsat, MD, u2, LS2, sunctx2);
    SetCVODE(mem3, f_gw,    MD, u3, LS3, sunctx3);
    SetCVODE(mem4, f_river, MD, u4, LS4, sunctx4);
    SetCVODE(mem5, f_lake,  MD, u5, LS5, sunctx5);
    
//    flag = CVodeSetMaxStep(mem1, max(MD->CS.MaxStep/4., 1.) );
//    check_flag(&flag, "CVodeSetMaxStep", 1);
    
    /* set start time */
    t = MD->CS.StartTime;
    double tnext = t;
    //CheckInput(MD, &CS);
    /* start solver in loops */
//    getSecond();
    MD->modelSummary(0);
    MD->debugData(fout->outpath);
    MD->gc.write(fout->Calib_bak);

    const auto bench_wall_start = std::chrono::steady_clock::now();
    double bench_forcing_s = 0.0;
    double bench_cvode_s = 0.0;
    double bench_io_s = 0.0;
    
//    FILE *fp1, *fp2, *fp3, *fp4;
//    fp1=fopen("y1.txt", "w");
//    fp2=fopen("y2.txt", "w");
//    fp3=fopen("y3.txt", "w");
//    fp4=fopen("y4.txt", "w");
    double t0 = t;
    const bool etSubstepEnabled =
        (MD->CS.ETStep > ZERO && MD->CS.ETStep + ZERO < MD->CS.SolverStep);
    for (int i = 0; i < MD->CS.NumSteps && !ierr; i++) {
        /* inner loops to next output points with ET step size control */
        tnext += MD->CS.SolverStep;
        while (t + ZERO < tnext) {
            tout = tnext;
            if (etSubstepEnabled) {
                tout = min(t + MD->CS.ETStep, tnext);
            }

            dt = tout - t;
            t0 = t;
            const auto forcing_start = std::chrono::steady_clock::now();
            MD->updateAllTimeSeries(t);
            MD->updateforcing(t);
            /* calculate Interception Storage */
            MD->ET(t, tout);
            bench_forcing_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - forcing_start).count();
            
            t = t0;
            MD->t0 = t0;
            MD->t1 = tout;
            Global2Sub(MD->NumEle, MD->NumRiv, MD->NumLake);
            if (etSubstepEnabled) {
                flag = CVodeSetStopTime(mem1, tout);
                check_flag(&flag, "CVodeSetStopTime", 1);
            }
            const auto cvode1_start = std::chrono::steady_clock::now();
            flag = CVode(mem1, tout, u1, &t, CV_NORMAL);
            bench_cvode_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - cvode1_start).count();
            check_flag(&flag, "CVode1 SURF", 1);
            
            t = t0;
            Global2Sub(MD->NumEle, MD->NumRiv, MD->NumLake);
            if (etSubstepEnabled) {
                flag = CVodeSetStopTime(mem2, tout);
                check_flag(&flag, "CVodeSetStopTime", 1);
            }
            const auto cvode2_start = std::chrono::steady_clock::now();
            flag = CVode(mem2, tout, u2, &t, CV_NORMAL);
            bench_cvode_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - cvode2_start).count();
            check_flag(&flag, "CVode2 UNSAT", 1);
            
            t = t0;
            Global2Sub(MD->NumEle, MD->NumRiv, MD->NumLake);
            if (etSubstepEnabled) {
                flag = CVodeSetStopTime(mem3, tout);
                check_flag(&flag, "CVodeSetStopTime", 1);
            }
            const auto cvode3_start = std::chrono::steady_clock::now();
            flag = CVode(mem3, tout, u3, &t, CV_NORMAL);
            bench_cvode_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - cvode3_start).count();
            check_flag(&flag, "CVode3 GW", 1);
            
            t = t0;
            Global2Sub(MD->NumEle, MD->NumRiv, MD->NumLake);
            if (etSubstepEnabled) {
                flag = CVodeSetStopTime(mem4, tout);
                check_flag(&flag, "CVodeSetStopTime", 1);
            }
            const auto cvode4_start = std::chrono::steady_clock::now();
            flag = CVode(mem4, tout, u4, &t, CV_NORMAL);
            bench_cvode_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - cvode4_start).count();
            check_flag(&flag, "CVode4 RIV", 1);
            
            if(lakeon && N5 > 0){
                t = t0;
                Global2Sub(MD->NumEle, MD->NumRiv, MD->NumLake);
                if (etSubstepEnabled) {
                    flag = CVodeSetStopTime(mem5, tout);
                    check_flag(&flag, "CVodeSetStopTime", 1);
                }
                const auto cvode5_start = std::chrono::steady_clock::now();
                flag = CVode(mem5, tout, u5, &t, CV_NORMAL);
                bench_cvode_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - cvode5_start).count();
                check_flag(&flag, "CVode5 LAKE", 1);
            }
        }
        {
            const auto io_start = std::chrono::steady_clock::now();
            MD->summary(u1, u2, u3, u4, u5);
            MD->CS.ExportResults(t);
            flag = MD->ScreenPrintu(t, i);
            MD->PrintInit(fout->Init_update, t);
            MD->flood->FloodWarning(t);
            bench_io_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - io_start).count();
        }
//        printVector(fp1, globalY, 0, N1, t);
//        printVector(fp2, globalY, N1, N2, t);
//        printVector(fp3, globalY, N1*2, N3, t);
//        printVector(fp4, globalY, N1*3, N4, t);
    }
//    fclose(fp1);
//    fclose(fp2);
//    fclose(fp3);
//    fclose(fp4);
    const double bench_wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - bench_wall_start).count();
    printf("\nBENCH_STATS wall_s=%.6f cvode_s=%.6f io_s=%.6f forcing_s=%.6f\n\n",
           bench_wall_s,
           bench_cvode_s,
           bench_io_s,
           bench_forcing_s);

    MD->modelSummary(1);
    /* Free memory */
    N_VDestroy(u1);
    N_VDestroy(u2);
    N_VDestroy(u3);
    N_VDestroy(u4);
    N_VDestroy(u5);
    
    N_VDestroy(du1);
    N_VDestroy(du2);
    N_VDestroy(du3);
    N_VDestroy(du4);
    N_VDestroy(du5);
    /* Free integrator memory */
    CVodeFree(&mem1);
    CVodeFree(&mem2);
    CVodeFree(&mem3);
    CVodeFree(&mem4);
    CVodeFree(&mem5);
    SUNLinSolFree(LS1);
    SUNLinSolFree(LS2);
    SUNLinSolFree(LS3);
    SUNLinSolFree(LS4);
    SUNLinSolFree(LS5);

    SUNContext_Free(&sunctx1);
    SUNContext_Free(&sunctx2);
    SUNContext_Free(&sunctx3);
    SUNContext_Free(&sunctx4);
    SUNContext_Free(&sunctx5);
    
    delete MD;
    return ret;
}

int SHUD(int argc, char *argv[]){
    CommandIn CLI;
    FileIn *fin = new FileIn;
    FileOut *fout = new FileOut;

    /* CUDA preconditioner toggle (default ON).
     * - Env: SHUD_CUDA_PRECOND=0/1 (also accepts true/false/on/off/yes/no)
     * - CLI: --precond / --no-precond (overrides env)
     */
    const char *precond_env = getenv("SHUD_CUDA_PRECOND");
    if (precond_env != NULL && precond_env[0] != '\0') {
        int enabled = global_precond_enabled;
        if (parseBoolEnvValue(precond_env, &enabled)) {
            global_precond_enabled = enabled;
        } else {
            fprintf(stderr,
                    "WARNING: invalid SHUD_CUDA_PRECOND='%s' (expect 0/1, on/off, true/false); using %d.\n",
                    precond_env,
                    global_precond_enabled);
        }
    }

    CLI.parse(argc, argv);
    CLI.setFileIO(fin, fout);

    if (global_backend == BACKEND_OMP) {
#ifndef _OPENMP_ON
        fprintf(stderr, "\nERROR: --backend omp requested, but this build does not enable OpenMP.\n\n");
        myexit(-1);
#endif
    } else if (global_backend == BACKEND_CUDA) {
#ifndef _CUDA_ON
        fprintf(stderr, "\nERROR: --backend cuda requested, but this build does not enable CUDA (NVECTOR_CUDA).\n\n");
        myexit(-1);
#endif
    }

    if(global_implicit_mode){
        SHUD(fin, fout);
    }else{
        SHUD_uncouple(fin, fout);
    }
    delete fin;
    delete fout;
    return 0;
}
