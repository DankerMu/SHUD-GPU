//  CommandIn.cpp
//
//  Created by Lele Shu on 9/29/18.
//  Copyright © 2018 Lele Shu. All rights reserved.
//

#include "CommandIn.hpp"
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <string.h>

static char *ltrim(char *s)
{
    if (s == NULL) {
        return NULL;
    }
    while (*s != '\0' && isspace(static_cast<unsigned char>(*s))) {
        s++;
    }
    return s;
}

static void rtrim(char *s)
{
    if (s == NULL) {
        return;
    }
    size_t n = strlen(s);
    while (n > 0 && isspace(static_cast<unsigned char>(s[n - 1]))) {
        s[n - 1] = '\0';
        n--;
    }
}

static bool streq_ci(const char *a, const char *b)
{
    if (a == NULL || b == NULL) {
        return false;
    }
    while (*a != '\0' && *b != '\0') {
        const unsigned char ca = static_cast<unsigned char>(*a);
        const unsigned char cb = static_cast<unsigned char>(*b);
        if (tolower(ca) != tolower(cb)) {
            return false;
        }
        a++;
        b++;
    }
    return (*a == '\0' && *b == '\0');
}

static bool parseOutputGroupsArg(const char *arg, int *out_mask)
{
    if (arg == NULL || out_mask == NULL) {
        return false;
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "%s", arg);
    char *p = ltrim(buf);
    rtrim(p);
    if (*p == '\0') {
        return false;
    }

    if (streq_ci(p, "all") || streq_ci(p, "full")) {
        *out_mask = OUTPUT_GROUP_ALL;
        return true;
    }
    if (streq_ci(p, "off") || streq_ci(p, "none")) {
        *out_mask = 0;
        return true;
    }

    int mask = 0;
    char *saveptr = NULL;
    for (char *tok = strtok_r(p, ",", &saveptr); tok != NULL; tok = strtok_r(NULL, ",", &saveptr)) {
        char *t = ltrim(tok);
        rtrim(t);
        if (*t == '\0') {
            continue;
        }
        if (streq_ci(t, "state")) {
            mask |= OUTPUT_GROUP_STATE;
        } else if (streq_ci(t, "flux")) {
            mask |= OUTPUT_GROUP_FLUX;
        } else if (streq_ci(t, "diag") || streq_ci(t, "diagnostic")) {
            mask |= OUTPUT_GROUP_DIAG;
        } else {
            return false;
        }
    }
    *out_mask = mask;
    return true;
}

void CommandIn::SHUD_help(const char *prog){
    printf ("\n\nUsage:\n");
    printf ("%s [-0fgv] [-C ClampPolicy] [-p project_file] [-c Calib_file] [-o output] [-n Num_Threads] [--backend cpu|omp|cuda] [--precond|--no-precond|--precond-auto] [--io <groups>] [--help] <project_name>\n\n", prog);
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
    printf ("          Auto-selection (default ON when binary defaults to cuda):\n");
    printf ("            Env SHUD_BACKEND_AUTO=0/1 enables/disables.\n");
    printf ("            Env NY_GPU_MIN sets CUDA threshold (default 100000).\n");
    printf ("            Override auto-selection with --backend cuda.\n");
    printf (" --precond Enable CVODE preconditioner (CUDA backend only; default ON for --backend cuda).\n");
    printf (" --no-precond Disable CVODE preconditioner.\n");
    printf (" --precond-auto Auto-select CVODE preconditioner (CUDA backend only).\n");
    printf ("          Env override: SHUD_CUDA_PRECOND=0/1/auto (default 1).\n");
    printf ("          Auto threshold: NY_CUDA_PRECOND_MIN (default 100000).\n");
    printf (" --io Output groups: all|full|off|none|state,flux,diag\n");
    printf ("          Examples: --io off, --io state, --io state,flux\n");
    printf (" --help Print this message and exit.\n");
}

void CommandIn::parse(int argc, char **argv){
    if(argc<=1){
        SHUD_help(argv[0]);
        myexit(ERRSUCCESS);
    }
    global_backend_cli_set = 0;

    static struct option long_options[] = {
        {"backend", required_argument, NULL, 1},
        {"precond", no_argument, NULL, 2},
        {"no-precond", no_argument, NULL, 3},
        {"precond-auto", no_argument, NULL, 4},
        {"io", required_argument, NULL, 5},
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
                global_backend_cli_set = 1;
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
                global_precond_mode = PRECOND_MODE_ON;
                global_precond_enabled = 1;
                break;
            case 3:
                global_precond_mode = PRECOND_MODE_OFF;
                global_precond_enabled = 0;
                break;
            case 4:
                global_precond_mode = PRECOND_MODE_AUTO;
                break;
            case 5: {
                int mask = OUTPUT_GROUP_ALL;
                if (!parseOutputGroupsArg(optarg, &mask)) {
                    fprintf(stderr,
                            "ERROR: invalid --io '%s' (expect all|full|off|none|state,flux,diag)\n",
                            optarg);
                    myexit(-1);
                }
                global_output_groups = mask;
                break;
            }
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
