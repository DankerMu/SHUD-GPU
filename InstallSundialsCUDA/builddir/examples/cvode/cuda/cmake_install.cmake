# Install script for directory: /mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/cuda

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/root/sundials")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/cuda/cvAdvDiff_kry_cuda.cu;/root/sundials/example/cvode/cuda/cvAdvDiff_kry_cuda.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/cuda" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu;/root/sundials/example/cvode/cuda/cvAdvDiff_kry_cuda_managed.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/cuda" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/cuda/cvAdvDiff_diag_cuda.cu;/root/sundials/example/cvode/cuda/cvAdvDiff_diag_cuda_0_0.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/cuda" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/cuda/cvAdvDiff_diag_cuda_0_0.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/cuda/cvAdvDiff_diag_cuda.cu;/root/sundials/example/cvode/cuda/cvAdvDiff_diag_cuda_0_1.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/cuda" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/cuda/cvAdvDiff_diag_cuda_0_1.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/cuda/cvAdvDiff_diag_cuda.cu;/root/sundials/example/cvode/cuda/cvAdvDiff_diag_cuda_1_1.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/cuda" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/cuda/cvAdvDiff_diag_cuda_1_1.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/cuda/README")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/cuda" TYPE FILE FILES "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/cuda/README")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/cuda/CMakeLists.txt")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/cuda" TYPE FILE FILES "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/builddir/examples/cvode/cuda/CMakeLists.txt")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/cuda/Makefile")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/cuda" TYPE FILE RENAME "Makefile" FILES "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/builddir/examples/cvode/cuda/Makefile_ex")
endif()

