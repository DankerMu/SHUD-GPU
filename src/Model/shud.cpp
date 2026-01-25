#include <stdio.h>
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
int lakeon = 0; /* Whether lake module ON(1), OFF(0) */
int CLAMP_POLICY = 1; /* Whether to clamp state to non-negative values */
int CLAMP_POLICY_CLI_SET = 0; /* Whether CLAMP_POLICY is overridden by CLI (-C) */
using namespace std;

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
             * results, keep NVECTOR_OPENMP math reductions single-threaded.
             */
            const int nvec_threads = 1;
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
//    f(t, udata, du, MD); /* Initialized the status */
    const bool etSubstepEnabled =
        (MD->CS.ETStep > ZERO && MD->CS.ETStep + ZERO < MD->CS.SolverStep);
    for (int i = 0; i < MD->CS.NumSteps && !ierr; i++) {
        printDY(MD->file_debug);
#ifdef DEBUG
        printDY(MD->file_debug);
#endif
        flag = MD->ScreenPrint(t, i);
        MD->PrintInit(fout->Init_update, t);
        /* inner loops to next output points with ET step size control */
        tnext += MD->CS.SolverStep;
        while (t + ZERO < tnext) {
            double tout = tnext;
            if (etSubstepEnabled) {
                tout = min(t + MD->CS.ETStep, tnext);
            }

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
            if (dummy_mode) {
                t = tout; /* dummy mode only. */
            } else {
                if (etSubstepEnabled) {
                    flag = CVodeSetStopTime(mem, tout);
                    check_flag(&flag, "CVodeSetStopTime", 1);
                }
                flag = CVode(mem, tout, udata, &t, CV_NORMAL);
                check_flag(&flag, "CVode", 1);
            }
        }
        //            CVODEstatus(mem, udata, t);
#ifdef _CUDA_ON
        MD->gpuSyncStateFromDevice(udata);
        MD->gpuSyncDiagnosticsFromDevice(udata);
#endif
        MD->summary(udata);
        MD->CS.ExportResults(t);
        MD->flood->FloodWarning(t);
    }
    MD->ScreenPrint(t, MD->CS.NumSteps);
    MD->PrintInit(fout->Init_update, t);

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

    if (global_backend == BACKEND_CUDA) {
        fprintf(stderr,
                "\nERROR: --backend cuda is not supported in uncoupled mode (-g). "
                "Please run coupled mode or use --backend cpu/omp.\n\n");
        myexit(-1);
    }

    if (global_backend == BACKEND_OMP) {
#ifdef _OPENMP_ON
        omp_set_num_threads(nthreads);
        screeninfo("\nBackend: omp (NVECTOR_OPENMP). Threads = %d\n", nthreads);
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
        u1 = N_VNew_OpenMP(N1, nthreads, sunctx1);
        u2 = N_VNew_OpenMP(N2, nthreads, sunctx2);
        u3 = N_VNew_OpenMP(N3, nthreads, sunctx3);
        u4 = N_VNew_OpenMP(N4, nthreads, sunctx4);
        u5 = N_VNew_OpenMP(N5, nthreads, sunctx5);

        du1 = N_VNew_OpenMP(N1, nthreads, sunctx1);
        du2 = N_VNew_OpenMP(N2, nthreads, sunctx2);
        du3 = N_VNew_OpenMP(N3, nthreads, sunctx3);
        du4 = N_VNew_OpenMP(N4, nthreads, sunctx4);
        du5 = N_VNew_OpenMP(N5, nthreads, sunctx5);
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
            MD->updateAllTimeSeries(t);
            MD->updateforcing(t);
            /* calculate Interception Storage */
            MD->ET(t, tout);
            
            t = t0;
            MD->t0 = t0;
            MD->t1 = tout;
            Global2Sub(MD->NumEle, MD->NumRiv, MD->NumLake);
            if (etSubstepEnabled) {
                flag = CVodeSetStopTime(mem1, tout);
                check_flag(&flag, "CVodeSetStopTime", 1);
            }
            flag = CVode(mem1, tout, u1, &t, CV_NORMAL);
            check_flag(&flag, "CVode1 SURF", 1);
            
            t = t0;
            Global2Sub(MD->NumEle, MD->NumRiv, MD->NumLake);
            if (etSubstepEnabled) {
                flag = CVodeSetStopTime(mem2, tout);
                check_flag(&flag, "CVodeSetStopTime", 1);
            }
            flag = CVode(mem2, tout, u2, &t, CV_NORMAL);
            check_flag(&flag, "CVode2 UNSAT", 1);
            
            t = t0;
            Global2Sub(MD->NumEle, MD->NumRiv, MD->NumLake);
            if (etSubstepEnabled) {
                flag = CVodeSetStopTime(mem3, tout);
                check_flag(&flag, "CVodeSetStopTime", 1);
            }
            flag = CVode(mem3, tout, u3, &t, CV_NORMAL);
            check_flag(&flag, "CVode3 GW", 1);
            
            t = t0;
            Global2Sub(MD->NumEle, MD->NumRiv, MD->NumLake);
            if (etSubstepEnabled) {
                flag = CVodeSetStopTime(mem4, tout);
                check_flag(&flag, "CVodeSetStopTime", 1);
            }
            flag = CVode(mem4, tout, u4, &t, CV_NORMAL);
            check_flag(&flag, "CVode4 RIV", 1);
            
            if(lakeon && N5 > 0){
                t = t0;
                Global2Sub(MD->NumEle, MD->NumRiv, MD->NumLake);
                if (etSubstepEnabled) {
                    flag = CVodeSetStopTime(mem5, tout);
                    check_flag(&flag, "CVodeSetStopTime", 1);
                }
                flag = CVode(mem5, tout, u5, &t, CV_NORMAL);
                check_flag(&flag, "CVode5 LAKE", 1);
            }
        }
        MD->summary(u1, u2, u3, u4, u5);
        MD->CS.ExportResults(t);
        flag = MD->ScreenPrintu(t, i);
        MD->PrintInit(fout->Init_update, t);
//        printVector(fp1, globalY, 0, N1, t);
//        printVector(fp2, globalY, N1, N2, t);
//        printVector(fp3, globalY, N1*2, N3, t);
//        printVector(fp4, globalY, N1*3, N4, t);
        MD->flood->FloodWarning(t);
    }
//    fclose(fp1);
//    fclose(fp2);
//    fclose(fp3);
//    fclose(fp4);
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
