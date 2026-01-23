# -----------------------------------------------------------------
# Version: 1.0
# Date: Nov 2019
# Makefile for SHUD v 1.0
# -----------------------------------------------------------------
# Programmer: Lele Shu (lele.shu@gmail.com)
# SHUD model is a heritage of Penn State Integrated Hydrologic Model (PIHM).
# -----------------------------------------------------------------
#  Prerequisite:
#  1 install sundials 5.0+ via https://computation.llnl.gov/projects/sundials/sundials-software.
#  2 If parallel-computing is prefered, please install OpenMP.
#	 For mac: 
#	  		brew install llvm clang
#			brew install libomp
#			compile flags for OpenMP: 
#				-Xpreprocessor -fopenmp -lomp
#			Library/Include paths:
#				-L/usr/local/opt/libomp/lib 
#				-I/usr/local/opt/libomp/include
#  3 For CUDA build (Linux / CUDA machine only):
#       - Install CUDA Toolkit (nvcc) and a CUDA-enabled SUNDIALS build (NVECTOR_CUDA).
#       - Use ./configure_cuda to build/install SUNDIALS with ENABLE_CUDA=ON.
#       - macOS has no CUDA environment; `make -n shud_cuda` is for syntax dry-run only.
#			
# -----------------------------------------------------------------
# Configure this File:
# 1 Path of SUNDIALS_DIR. [CRITICAL]
# 2 Path of OpenMP if parallel is preffered.
# 3 Path of SRC_DIR, default is "SRC_DIR = ."
# 4 Path of BUILT_DIR, default is "BUILT_DIR = ."
# -----------------------------------------------------------------
SUNDIALS_DIR = $(HOME)/sundials
# SUNDIALS_DIR = /usr/local/sundials


SHELL = /bin/sh
BUILDDIR = .
SRC_DIR = src

LIB_SYS = /usr/local/lib/
INC_OMP = /usr/local/opt/libomp/include
LIB_OMP = /usr/local/opt/libomp/lib
LIB_SUN = ${SUNDIALS_DIR}/lib
CUDA_HOME ?= /usr/local/cuda
INC_CUDA ?= ${CUDA_HOME}/include
LIB_CUDA ?= ${CUDA_HOME}/lib64

INC_MPI = /usr/local/opt/open-mpi

TARGET_EXEC     = ${BUILDDIR}/shud
TARGET_OMP      = ${BUILDDIR}/shud_omp
TARGET_DEBUG    = ${BUILDDIR}/shud_debug
TARGET_CUDA     = ${BUILDDIR}/shud_cuda

MAIN_shud 		= ${SRC_DIR}/main.cpp
MAIN_OMP 		= ${SRC_DIR}/main.cpp
MAIN_DEBUG 		= ${SRC_DIR}/main.cpp

# If compile on Cluster
# CC       = g++
# MPICC    = mpic++
# LK_OMP   = -fopenmp -lsundials_nvecopenmp
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SUNDIALS_DIR}/lib

CC       = /usr/bin/g++
MPICC    = /usr/local/bin/mpic++
NVCC     ?= nvcc
CFLAGS   = -O3 -g  -std=c++14
#STCFLAG     = -static

SRC    	= ${SRC_DIR}/classes/*.cpp \
		  ${SRC_DIR}/ModelData/*.cpp \
		  ${SRC_DIR}/Model/*.cpp \
		  ${SRC_DIR}/GPU/*.cpp \
		  ${SRC_DIR}/Equations/*.cpp

SRC_H	= ${SRC_DIR}/classes/*.hpp \
		  ${SRC_DIR}/ModelData/*.hpp \
		  ${SRC_DIR}/Model/*.hpp \
		  ${SRC_DIR}/Equations/*.hpp \
		  ${SRC_DIR}/GPU/*.hpp


INCLUDES = -I ${SUNDIALS_DIR}/include \
		   -I ${INC_OMP} \
		   -I ${SRC_DIR}/Model \
		   -I ${SRC_DIR}/ModelData \
		   -I ${SRC_DIR}/classes \
		   -I ${SRC_DIR}/Equations \
		   -I ${SRC_DIR}/GPU 

		  
LIBRARIES = -L ${LIB_OMP} \
			-L ${LIB_SUN} \
			-L ${LIB_SYS}

RPATH = '-Wl,-rpath,${LIB_SUN}' 
RPATH_CUDA = '-Wl,-rpath,${LIB_SUN}' '-Wl,-rpath,${LIB_CUDA}'

LK_FLAGS = -lm -lsundials_cvode -lsundials_nvecserial
LK_OMP	= -fopenmp -lsundials_nvecopenmp
LK_CUDA  = -lm -lsundials_cvode -lsundials_nvecserial -lsundials_nveccuda -lsundials_sunmemcuda -lcudart
LK_DYLN = "LD_LIBRARY_PATH=${LIB_SUN}"

# Default supported GPU architectures (sm_70/75/80/86).
# Override at build time if needed, e.g.:
#   make shud_cuda CUDA_GENCODE='-gencode arch=compute_80,code=sm_80'
CUDA_GENCODE ?= -gencode arch=compute_70,code=sm_70 \
			   -gencode arch=compute_75,code=sm_75 \
			   -gencode arch=compute_80,code=sm_80 \
			   -gencode arch=compute_86,code=sm_86

# CUDA sources are optional; this expands to empty if src/GPU/*.cu does not exist yet.
CUDA_SRC = $(wildcard ${SRC_DIR}/GPU/*.cu)

.PHONY: all check help cvode CVODE shud SHUD shud_omp shud_cuda clean

all:
	$(MAKE) clean
	$(MAKE) shud
	@echo
check:
	ls ${SUNDIALS_DIR}
	ls ${SUNDIALS_DIR}/lib
	./shud
	@echo
help:
	@(echo)
	@echo "Usage:"
	@(echo '       make all	    	- clean and make shud')
	@(echo '       make cvode	    - install SUNDIALS/CVODE to ~/sundials')
	@(echo '       make shud     	- make shud executable')
	@(echo '       make shud_omp    - make shud_omp with OpenMP support')
	@(echo '       make shud_cuda   - make shud_cuda with CUDA (NVECTOR_CUDA) support')
	@(echo)
	@(echo '       make clean    	- remove all executable files')
	@(echo)
cvode CVODE:
	@echo '...Install SUNDIALS/CVODE for your ...'
	chmod +x configure
	./configure
	@echo 

shud SHUD: ${MAIN_shud} $(SRC) $(SRC_H)
	@echo '...Compiling shud ...'
	@echo  $(CC) $(CFLAGS) ${STCFLAG} ${INCLUDES} ${LIBRARIES} ${RPATH} -o ${TARGET_EXEC} ${MAIN_shud} $(SRC)  $(LK_FLAGS)
	@echo
	@echo
	 $(CC) $(CFLAGS) ${INCLUDES} ${STCFLAG} ${LIBRARIES} ${RPATH} -o ${TARGET_EXEC} ${MAIN_shud} $(SRC)  $(LK_FLAGS)
	@echo
	@echo
	@echo " ${TARGET_EXEC} is compiled successfully!"
	@echo

shud_omp: ${MAIN_OMP}  $(SRC) $(SRC_H)
	@echo '...Compiling shud_OpenMP ...'
	@echo $(CC) $(CFLAGS) ${STCFLAG} ${RPATH} -D_OPENMP_ON ${INCLUDES} ${LIBRARIES} -o ${TARGET_OMP}   ${MAIN_OMP} $(SRC)  $(LK_FLAGS) $(LK_OMP)
	@echo
	@echo
	$(CC) $(CFLAGS)  ${STCFLAG} ${RPATH} -D_OPENMP_ON ${INCLUDES} ${LIBRARIES} -o ${TARGET_OMP}   ${MAIN_OMP} $(SRC)  $(LK_FLAGS) $(LK_OMP)
	@echo
	@echo " ${TARGET_OMP} is compiled successfully!"
	@echo
	@echo

shud_cuda: ${MAIN_shud} $(SRC) $(SRC_H) $(CUDA_SRC)
	@echo '...Compiling shud_CUDA (NVECTOR_CUDA) ...'
	@echo $(NVCC) $(CFLAGS) ${STCFLAG} $(CUDA_GENCODE) -D_CUDA_ON ${INCLUDES} -I ${INC_CUDA} ${LIBRARIES} -L ${LIB_CUDA} ${RPATH_CUDA} -o ${TARGET_CUDA} ${MAIN_shud} $(SRC) $(CUDA_SRC) $(LK_CUDA)
	@echo
	@echo
	$(NVCC) $(CFLAGS) ${STCFLAG} $(CUDA_GENCODE) -D_CUDA_ON ${INCLUDES} -I ${INC_CUDA} ${LIBRARIES} -L ${LIB_CUDA} ${RPATH_CUDA} -o ${TARGET_CUDA} ${MAIN_shud} $(SRC) $(CUDA_SRC) $(LK_CUDA)
	@echo
	@echo " ${TARGET_CUDA} is compiled successfully!"
	@echo

clean:
	@echo "Cleaning ... "
	@echo
	@echo "  rm -f *.o"
	@rm -f *.o
	
	@echo "  rm -f ${TARGET_EXEC}"
	@rm -f ${TARGET_EXEC}
	
	@echo "  rm -f ${TARGET_OMP}"
	@rm -f ${TARGET_OMP}

	@echo "  rm -f ${TARGET_CUDA}"
	@rm -f ${TARGET_CUDA}
		
	@echo
	@echo "Done."
	@echo

