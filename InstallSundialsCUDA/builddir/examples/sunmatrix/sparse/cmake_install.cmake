# Install script for directory: /mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse

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
   "/root/sundials/example/sunmatrix/sparse/test_sunmatrix_sparse.c;/root/sundials/example/sunmatrix/sparse/test_sunmatrix.c;/root/sundials/example/sunmatrix/sparse/test_sunmatrix.h;/root/sundials/example/sunmatrix/sparse/sundials_matrix.c;/root/sundials/example/sunmatrix/sparse/sundials_nvector.c")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/sunmatrix/sparse" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/test_sunmatrix_sparse.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/../test_sunmatrix.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/../test_sunmatrix.h"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/src/sundials/sundials_matrix.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/src/sundials/sundials_nvector.c"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/sunmatrix/sparse/test_sunmatrix_sparse.c;/root/sundials/example/sunmatrix/sparse/test_sunmatrix.c;/root/sundials/example/sunmatrix/sparse/test_sunmatrix.h;/root/sundials/example/sunmatrix/sparse/sundials_matrix.c;/root/sundials/example/sunmatrix/sparse/sundials_nvector.c")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/sunmatrix/sparse" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/test_sunmatrix_sparse.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/../test_sunmatrix.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/../test_sunmatrix.h"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/src/sundials/sundials_matrix.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/src/sundials/sundials_nvector.c"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/sunmatrix/sparse/test_sunmatrix_sparse.c;/root/sundials/example/sunmatrix/sparse/test_sunmatrix.c;/root/sundials/example/sunmatrix/sparse/test_sunmatrix.h;/root/sundials/example/sunmatrix/sparse/sundials_matrix.c;/root/sundials/example/sunmatrix/sparse/sundials_nvector.c")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/sunmatrix/sparse" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/test_sunmatrix_sparse.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/../test_sunmatrix.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/../test_sunmatrix.h"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/src/sundials/sundials_matrix.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/src/sundials/sundials_nvector.c"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/sunmatrix/sparse/test_sunmatrix_sparse.c;/root/sundials/example/sunmatrix/sparse/test_sunmatrix.c;/root/sundials/example/sunmatrix/sparse/test_sunmatrix.h;/root/sundials/example/sunmatrix/sparse/sundials_matrix.c;/root/sundials/example/sunmatrix/sparse/sundials_nvector.c")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/sunmatrix/sparse" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/test_sunmatrix_sparse.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/../test_sunmatrix.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/../test_sunmatrix.h"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/src/sundials/sundials_matrix.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/src/sundials/sundials_nvector.c"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/sunmatrix/sparse/test_sunmatrix_sparse.c;/root/sundials/example/sunmatrix/sparse/test_sunmatrix.c;/root/sundials/example/sunmatrix/sparse/test_sunmatrix.h;/root/sundials/example/sunmatrix/sparse/sundials_matrix.c;/root/sundials/example/sunmatrix/sparse/sundials_nvector.c")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/sunmatrix/sparse" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/test_sunmatrix_sparse.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/../test_sunmatrix.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/../test_sunmatrix.h"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/src/sundials/sundials_matrix.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/src/sundials/sundials_nvector.c"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/sunmatrix/sparse/test_sunmatrix_sparse.c;/root/sundials/example/sunmatrix/sparse/test_sunmatrix.c;/root/sundials/example/sunmatrix/sparse/test_sunmatrix.h;/root/sundials/example/sunmatrix/sparse/sundials_matrix.c;/root/sundials/example/sunmatrix/sparse/sundials_nvector.c")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/sunmatrix/sparse" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/test_sunmatrix_sparse.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/../test_sunmatrix.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/sunmatrix/sparse/../test_sunmatrix.h"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/src/sundials/sundials_matrix.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/src/sundials/sundials_nvector.c"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/sunmatrix/sparse/CMakeLists.txt")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/sunmatrix/sparse" TYPE FILE FILES "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/builddir/examples/sunmatrix/sparse/CMakeLists.txt")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/sunmatrix/sparse/Makefile")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/sunmatrix/sparse" TYPE FILE RENAME "Makefile" FILES "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/builddir/examples/sunmatrix/sparse/Makefile_ex")
endif()

