# Install script for directory: /mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial

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
   "/root/sundials/example/cvode/serial/cvAdvDiff_bnd.c;/root/sundials/example/cvode/serial/cvAdvDiff_bnd.out;/root/sundials/example/cvode/serial/cvAdvDiff_bndL.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvAdvDiff_bnd.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvAdvDiff_bnd.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvAdvDiff_bndL.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvAnalytic_mels.c;/root/sundials/example/cvode/serial/cvAnalytic_mels.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvAnalytic_mels.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvAnalytic_mels.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvDirectDemo_ls.c;/root/sundials/example/cvode/serial/cvDirectDemo_ls.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvDirectDemo_ls.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvDirectDemo_ls.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvDisc_dns.c;/root/sundials/example/cvode/serial/cvDisc_dns.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvDisc_dns.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvDisc_dns.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvDiurnal_kry_bp.c;/root/sundials/example/cvode/serial/cvDiurnal_kry_bp.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvDiurnal_kry_bp.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvDiurnal_kry_bp.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvDiurnal_kry.c;/root/sundials/example/cvode/serial/cvDiurnal_kry.out;/root/sundials/example/cvode/serial/cvDiurnal_kry_bp.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvDiurnal_kry.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvDiurnal_kry.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvDiurnal_kry_bp.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvKrylovDemo_ls.c;/root/sundials/example/cvode/serial/cvKrylovDemo_ls.out;/root/sundials/example/cvode/serial/cvKrylovDemo_ls_0_1.out;/root/sundials/example/cvode/serial/cvKrylovDemo_ls_1.out;/root/sundials/example/cvode/serial/cvKrylovDemo_ls_2.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls_0_1.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls_1.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls_2.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvKrylovDemo_ls.c;/root/sundials/example/cvode/serial/cvKrylovDemo_ls.out;/root/sundials/example/cvode/serial/cvKrylovDemo_ls_0_1.out;/root/sundials/example/cvode/serial/cvKrylovDemo_ls_1.out;/root/sundials/example/cvode/serial/cvKrylovDemo_ls_2.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls_0_1.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls_1.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls_2.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvKrylovDemo_ls.c;/root/sundials/example/cvode/serial/cvKrylovDemo_ls.out;/root/sundials/example/cvode/serial/cvKrylovDemo_ls_0_1.out;/root/sundials/example/cvode/serial/cvKrylovDemo_ls_1.out;/root/sundials/example/cvode/serial/cvKrylovDemo_ls_2.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls_0_1.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls_1.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_ls_2.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvKrylovDemo_prec.c;/root/sundials/example/cvode/serial/cvKrylovDemo_prec.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_prec.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvKrylovDemo_prec.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvParticle_dns.c;/root/sundials/example/cvode/serial/cvParticle_dns.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvParticle_dns.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvParticle_dns.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvPendulum_dns.c;/root/sundials/example/cvode/serial/cvPendulum_dns.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvPendulum_dns.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvPendulum_dns.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvRoberts_dns.c;/root/sundials/example/cvode/serial/cvRoberts_dns.out;/root/sundials/example/cvode/serial/cvRoberts_dnsL.out;/root/sundials/example/cvode/serial/cvRoberts_dns_constraints.out;/root/sundials/example/cvode/serial/cvRoberts_dns_negsol.out;/root/sundials/example/cvode/serial/cvRoberts_dns_uw.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvRoberts_dns.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvRoberts_dns.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvRoberts_dnsL.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvRoberts_dns_constraints.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvRoberts_dns_negsol.out"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvRoberts_dns_uw.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvRoberts_dns_constraints.c;/root/sundials/example/cvode/serial/cvRoberts_dns_constraints.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvRoberts_dns_constraints.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvRoberts_dns_constraints.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvRoberts_dns_negsol.c;/root/sundials/example/cvode/serial/cvRoberts_dns_negsol.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvRoberts_dns_negsol.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvRoberts_dns_negsol.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/cvRoberts_dns_uw.c;/root/sundials/example/cvode/serial/cvRoberts_dns_uw.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvRoberts_dns_uw.c"
    "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/cvRoberts_dns_uw.out"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/README")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/README")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/plot_cvParticle.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/plot_cvParticle.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/plot_cvPendulum.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/src/cvode-6.0.0/examples/cvode/serial/plot_cvPendulum.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/CMakeLists.txt")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE FILES "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/builddir/examples/cvode/serial/CMakeLists.txt")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/sundials/example/cvode/serial/Makefile")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/sundials/example/cvode/serial" TYPE FILE RENAME "Makefile" FILES "/mnt/sdc/SHUD-GPU/InstallSundialsCUDA/builddir/examples/cvode/serial/Makefile_ex")
endif()

