//  CommandIn.cpp
//
//  Created by Lele Shu on 9/29/18.
//  Copyright © 2018 Lele Shu. All rights reserved.
//

#include "CommandIn.hpp"
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
void CommandIn::SHUD_help(const char *prog){
    printf ("\n\nUsage:\n");
    printf ("%s [-0fgv] [-C ClampPolicy] [-p project_file] [-c Calib_file] [-o output] [-n Num_Threads] [--backend cpu|omp|cuda] [--precond|--no-precond] [--help] <project_name>\n\n", prog);
    printf (" -0 Dummy simulation. Load input and write output, but no calculation.\n");
    printf (" -f fflush for each time interval. fflush export data frequently, but slow down performance on cluster.\n");
    printf (" -g Sequential coupled Surface-Unsaturated-Saturated-River mode.\n");
    printf (" -v Dummy simulation. Load input and write output, but no calculation.\n");
    printf (" -C ClampPolicy for non-negative state truncation (0=OFF, 1=ON). Default is 1.\n");
    printf (" -c Calibration file (.cfg.calib). \n");
    printf (" -o output folder. Default is output/projname.out\n");
    printf (" -p projectfile, which includes the path to input files and output path.\n");
    printf (" -n Number of threads to run with OpenMP. \n");
    printf (" --backend Runtime backend selection: cpu, omp, cuda.\n");
    printf ("          Default depends on binary: shud→cpu, shud_omp→omp, shud_cuda→cuda.\n");
    printf (" --precond Enable CVODE preconditioner (CUDA backend only; default ON for --backend cuda).\n");
    printf (" --no-precond Disable CVODE preconditioner.\n");
    printf (" --help Print this message and exit.\n");
}

void CommandIn::parse(int argc, char **argv){
    if(argc<=1){
        SHUD_help(argv[0]);
        myexit(ERRSUCCESS);
    }

    static struct option long_options[] = {
        {"backend", required_argument, NULL, 1},
        {"precond", no_argument, NULL, 2},
        {"no-precond", no_argument, NULL, 3},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int option_index = 0;

    opterr = 0;
    while ((c = getopt_long(argc, argv, "0fgvC:c:e:n:o:p:h", long_options, &option_index)) != -1){
        switch (c){
            case '0':
                dummy_mode = 1;
                break;
            case 'f':
                global_fflush_mode = 1;
                break;
            case 'g':
                global_implicit_mode = 0;
                break;
            case 'v':
                global_verbose_mode = 1;
                break;
            case 'C': {
                char *endptr = NULL;
                errno = 0;
                const long v = strtol(optarg, &endptr, 10);
                if (errno == 0 && endptr != NULL && *endptr == '\0' && (v == 0 || v == 1)) {
                    CLAMP_POLICY = (int)v;
                    CLAMP_POLICY_CLI_SET = 1;
                } else {
                    fprintf(stderr,
                            "WARNING: invalid ClampPolicy '%s' (expect 0/1); using default %d.\n",
                            optarg,
                            CLAMP_POLICY);
                }
                break;
            }
            case 'c':
                strcpy(calibfile, optarg);
                break;
            case 'e':
                strcpy(dir_cmaes, optarg);
                break;
            case 'o':
                strcpy(outpath, optarg);
                iout = 1;
                break;
            case 'n':
                n_lambda = atoi(optarg) ;
                break;
            case 'p':
                strcpy(prjfile, optarg);
                iprj = 1;
                break;
            case 'h':
                SHUD_help(argv[0]);
                myexit(ERRSUCCESS);
                break;
            case 1:
                if (strcmp(optarg, "cpu") == 0) {
                    global_backend = BACKEND_CPU;
                } else if (strcmp(optarg, "omp") == 0) {
                    global_backend = BACKEND_OMP;
                } else if (strcmp(optarg, "cuda") == 0) {
                    global_backend = BACKEND_CUDA;
                } else {
                    fprintf(stderr, "ERROR: invalid --backend '%s' (expect cpu|omp|cuda).\n", optarg);
                    myexit(-1);
                }
                break;
            case 2:
                global_precond_enabled = 1;
                break;
            case 3:
                global_precond_enabled = 0;
                break;
            case '?':
                if (optopt == 'C') {
                    fprintf(stderr, "ERROR: option -%c requires an argument (0=OFF, 1=ON).\n", optopt);
                } else if (optopt == 'p' || optopt == 'c' || optopt == 'e' || optopt == 'n' || optopt == 'o') {
                    fprintf(stderr, "ERROR: option -%c requires an argument.\n", optopt);
                } else if (optopt == 0) {
                    const char *badopt = (optind > 0 && optind <= argc) ? argv[optind - 1] : NULL;
                    if (badopt != NULL && strcmp(badopt, "--backend") == 0) {
                        fprintf(stderr, "ERROR: option %s requires an argument (cpu|omp|cuda).\n", badopt);
                    } else if (badopt != NULL) {
                        fprintf(stderr, "ERROR: unknown option '%s'.\n", badopt);
                    } else {
                        fprintf(stderr, "ERROR: unknown option.\n");
                    }
                }
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown option character `\\x%x'.\n",
                             optopt);
                myexit(-1);
            default:
                break;
        }
    }
    if(iprj){
       // void
    }else{
        strcpy(prjname, argv[optind]);
    }
    
    
#ifdef DEBUG
    printf("\t\t\t * Debug mode enable.\n");
#endif
    
#ifdef _OPENMP_ON
    printf("openMP: ON\n");
    printf("\t\t * openMP enabled. Maximum Threads = %d\n", omp_get_max_threads());
#else
    printf("openMP: OFF\n");
    printf("\t\t * openMP disabled.\n");
#endif
}
CommandIn::CommandIn(){
    strcpy(prjname, "");
    strcpy(outpath, "");
    strcpy(inpath, "");
    strcpy(prjfile, "");
    strcpy(calibfile, "");
    strcpy(dir_cmaes, "cmaes");
}
void CommandIn::setFileIO(FileIn *fin, FileOut *fout){
    if(iprj){
        fin->readProject(prjfile);
    }else{
        sprintf(inpath, "input/%s", prjname);
        fin->setInFilePath(inpath, prjname, n_lambda);
        if (iout){
            fin->setOutpath(outpath);
        }
    }
    if(calibfile[0] != '\0' ){
        fin->setCalibFile(calibfile);
    }
    fout->setOutFilePath(fin->outpath, fin->projectname);
    fin->saveProject();
}
int CommandIn::getNumberThreads(){
    if(n_lambda < 0 ){
        return 0;
    }else{
        return n_lambda;
    }
}
