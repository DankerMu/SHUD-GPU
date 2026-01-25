#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SUNDIALS::generic_static" for configuration ""
set_property(TARGET SUNDIALS::generic_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::generic_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_generic.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::generic_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::generic_static "${_IMPORT_PREFIX}/lib/libsundials_generic.a" )

# Import target "SUNDIALS::generic_shared" for configuration ""
set_property(TARGET SUNDIALS::generic_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::generic_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_generic.so.6.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_generic.so.6"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::generic_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::generic_shared "${_IMPORT_PREFIX}/lib/libsundials_generic.so.6.0.0" )

# Import target "SUNDIALS::nvecserial_static" for configuration ""
set_property(TARGET SUNDIALS::nvecserial_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::nvecserial_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_nvecserial.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::nvecserial_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::nvecserial_static "${_IMPORT_PREFIX}/lib/libsundials_nvecserial.a" )

# Import target "SUNDIALS::nvecserial_shared" for configuration ""
set_property(TARGET SUNDIALS::nvecserial_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::nvecserial_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_nvecserial.so.6.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_nvecserial.so.6"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::nvecserial_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::nvecserial_shared "${_IMPORT_PREFIX}/lib/libsundials_nvecserial.so.6.0.0" )

# Import target "SUNDIALS::nvecmanyvector_static" for configuration ""
set_property(TARGET SUNDIALS::nvecmanyvector_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::nvecmanyvector_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_nvecmanyvector.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::nvecmanyvector_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::nvecmanyvector_static "${_IMPORT_PREFIX}/lib/libsundials_nvecmanyvector.a" )

# Import target "SUNDIALS::nvecmanyvector_shared" for configuration ""
set_property(TARGET SUNDIALS::nvecmanyvector_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::nvecmanyvector_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_nvecmanyvector.so.6.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_nvecmanyvector.so.6"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::nvecmanyvector_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::nvecmanyvector_shared "${_IMPORT_PREFIX}/lib/libsundials_nvecmanyvector.so.6.0.0" )

# Import target "SUNDIALS::nvecopenmp_static" for configuration ""
set_property(TARGET SUNDIALS::nvecopenmp_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::nvecopenmp_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_nvecopenmp.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::nvecopenmp_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::nvecopenmp_static "${_IMPORT_PREFIX}/lib/libsundials_nvecopenmp.a" )

# Import target "SUNDIALS::nvecopenmp_shared" for configuration ""
set_property(TARGET SUNDIALS::nvecopenmp_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::nvecopenmp_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_nvecopenmp.so.6.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_nvecopenmp.so.6"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::nvecopenmp_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::nvecopenmp_shared "${_IMPORT_PREFIX}/lib/libsundials_nvecopenmp.so.6.0.0" )

# Import target "SUNDIALS::nveccuda_static" for configuration ""
set_property(TARGET SUNDIALS::nveccuda_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::nveccuda_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C;CUDA"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_nveccuda.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::nveccuda_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::nveccuda_static "${_IMPORT_PREFIX}/lib/libsundials_nveccuda.a" )

# Import target "SUNDIALS::nveccuda_shared" for configuration ""
set_property(TARGET SUNDIALS::nveccuda_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::nveccuda_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_nveccuda.so.6.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_nveccuda.so.6"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::nveccuda_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::nveccuda_shared "${_IMPORT_PREFIX}/lib/libsundials_nveccuda.so.6.0.0" )

# Import target "SUNDIALS::sunmatrixband_static" for configuration ""
set_property(TARGET SUNDIALS::sunmatrixband_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunmatrixband_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunmatrixband.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunmatrixband_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunmatrixband_static "${_IMPORT_PREFIX}/lib/libsundials_sunmatrixband.a" )

# Import target "SUNDIALS::sunmatrixband_shared" for configuration ""
set_property(TARGET SUNDIALS::sunmatrixband_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunmatrixband_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunmatrixband.so.4.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_sunmatrixband.so.4"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunmatrixband_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunmatrixband_shared "${_IMPORT_PREFIX}/lib/libsundials_sunmatrixband.so.4.0.0" )

# Import target "SUNDIALS::sunmatrixdense_static" for configuration ""
set_property(TARGET SUNDIALS::sunmatrixdense_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunmatrixdense_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunmatrixdense.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunmatrixdense_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunmatrixdense_static "${_IMPORT_PREFIX}/lib/libsundials_sunmatrixdense.a" )

# Import target "SUNDIALS::sunmatrixdense_shared" for configuration ""
set_property(TARGET SUNDIALS::sunmatrixdense_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunmatrixdense_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunmatrixdense.so.4.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_sunmatrixdense.so.4"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunmatrixdense_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunmatrixdense_shared "${_IMPORT_PREFIX}/lib/libsundials_sunmatrixdense.so.4.0.0" )

# Import target "SUNDIALS::sunmatrixsparse_static" for configuration ""
set_property(TARGET SUNDIALS::sunmatrixsparse_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunmatrixsparse_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunmatrixsparse.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunmatrixsparse_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunmatrixsparse_static "${_IMPORT_PREFIX}/lib/libsundials_sunmatrixsparse.a" )

# Import target "SUNDIALS::sunmatrixsparse_shared" for configuration ""
set_property(TARGET SUNDIALS::sunmatrixsparse_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunmatrixsparse_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunmatrixsparse.so.4.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_sunmatrixsparse.so.4"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunmatrixsparse_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunmatrixsparse_shared "${_IMPORT_PREFIX}/lib/libsundials_sunmatrixsparse.so.4.0.0" )

# Import target "SUNDIALS::sunlinsolband_static" for configuration ""
set_property(TARGET SUNDIALS::sunlinsolband_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsolband_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolband.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsolband_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsolband_static "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolband.a" )

# Import target "SUNDIALS::sunlinsolband_shared" for configuration ""
set_property(TARGET SUNDIALS::sunlinsolband_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsolband_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolband.so.4.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_sunlinsolband.so.4.0.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsolband_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsolband_shared "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolband.so.4.0.0" )

# Import target "SUNDIALS::sunlinsoldense_static" for configuration ""
set_property(TARGET SUNDIALS::sunlinsoldense_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsoldense_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsoldense.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsoldense_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsoldense_static "${_IMPORT_PREFIX}/lib/libsundials_sunlinsoldense.a" )

# Import target "SUNDIALS::sunlinsoldense_shared" for configuration ""
set_property(TARGET SUNDIALS::sunlinsoldense_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsoldense_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsoldense.so.4.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_sunlinsoldense.so.4.0.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsoldense_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsoldense_shared "${_IMPORT_PREFIX}/lib/libsundials_sunlinsoldense.so.4.0.0" )

# Import target "SUNDIALS::sunlinsolpcg_static" for configuration ""
set_property(TARGET SUNDIALS::sunlinsolpcg_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsolpcg_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolpcg.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsolpcg_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsolpcg_static "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolpcg.a" )

# Import target "SUNDIALS::sunlinsolpcg_shared" for configuration ""
set_property(TARGET SUNDIALS::sunlinsolpcg_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsolpcg_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolpcg.so.4.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_sunlinsolpcg.so.4.0.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsolpcg_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsolpcg_shared "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolpcg.so.4.0.0" )

# Import target "SUNDIALS::sunlinsolspbcgs_static" for configuration ""
set_property(TARGET SUNDIALS::sunlinsolspbcgs_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsolspbcgs_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolspbcgs.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsolspbcgs_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsolspbcgs_static "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolspbcgs.a" )

# Import target "SUNDIALS::sunlinsolspbcgs_shared" for configuration ""
set_property(TARGET SUNDIALS::sunlinsolspbcgs_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsolspbcgs_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolspbcgs.so.4.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_sunlinsolspbcgs.so.4.0.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsolspbcgs_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsolspbcgs_shared "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolspbcgs.so.4.0.0" )

# Import target "SUNDIALS::sunlinsolspfgmr_static" for configuration ""
set_property(TARGET SUNDIALS::sunlinsolspfgmr_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsolspfgmr_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolspfgmr.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsolspfgmr_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsolspfgmr_static "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolspfgmr.a" )

# Import target "SUNDIALS::sunlinsolspfgmr_shared" for configuration ""
set_property(TARGET SUNDIALS::sunlinsolspfgmr_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsolspfgmr_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolspfgmr.so.4.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_sunlinsolspfgmr.so.4.0.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsolspfgmr_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsolspfgmr_shared "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolspfgmr.so.4.0.0" )

# Import target "SUNDIALS::sunlinsolspgmr_static" for configuration ""
set_property(TARGET SUNDIALS::sunlinsolspgmr_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsolspgmr_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolspgmr.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsolspgmr_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsolspgmr_static "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolspgmr.a" )

# Import target "SUNDIALS::sunlinsolspgmr_shared" for configuration ""
set_property(TARGET SUNDIALS::sunlinsolspgmr_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsolspgmr_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolspgmr.so.4.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_sunlinsolspgmr.so.4.0.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsolspgmr_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsolspgmr_shared "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolspgmr.so.4.0.0" )

# Import target "SUNDIALS::sunlinsolsptfqmr_static" for configuration ""
set_property(TARGET SUNDIALS::sunlinsolsptfqmr_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsolsptfqmr_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolsptfqmr.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsolsptfqmr_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsolsptfqmr_static "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolsptfqmr.a" )

# Import target "SUNDIALS::sunlinsolsptfqmr_shared" for configuration ""
set_property(TARGET SUNDIALS::sunlinsolsptfqmr_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunlinsolsptfqmr_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolsptfqmr.so.4.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_sunlinsolsptfqmr.so.4.0.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunlinsolsptfqmr_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunlinsolsptfqmr_shared "${_IMPORT_PREFIX}/lib/libsundials_sunlinsolsptfqmr.so.4.0.0" )

# Import target "SUNDIALS::sunnonlinsolnewton_static" for configuration ""
set_property(TARGET SUNDIALS::sunnonlinsolnewton_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunnonlinsolnewton_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunnonlinsolnewton.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunnonlinsolnewton_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunnonlinsolnewton_static "${_IMPORT_PREFIX}/lib/libsundials_sunnonlinsolnewton.a" )

# Import target "SUNDIALS::sunnonlinsolnewton_shared" for configuration ""
set_property(TARGET SUNDIALS::sunnonlinsolnewton_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunnonlinsolnewton_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunnonlinsolnewton.so.3.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_sunnonlinsolnewton.so.3.0.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunnonlinsolnewton_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunnonlinsolnewton_shared "${_IMPORT_PREFIX}/lib/libsundials_sunnonlinsolnewton.so.3.0.0" )

# Import target "SUNDIALS::sunnonlinsolfixedpoint_static" for configuration ""
set_property(TARGET SUNDIALS::sunnonlinsolfixedpoint_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunnonlinsolfixedpoint_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunnonlinsolfixedpoint.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunnonlinsolfixedpoint_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunnonlinsolfixedpoint_static "${_IMPORT_PREFIX}/lib/libsundials_sunnonlinsolfixedpoint.a" )

# Import target "SUNDIALS::sunnonlinsolfixedpoint_shared" for configuration ""
set_property(TARGET SUNDIALS::sunnonlinsolfixedpoint_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::sunnonlinsolfixedpoint_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_sunnonlinsolfixedpoint.so.3.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_sunnonlinsolfixedpoint.so.3.0.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::sunnonlinsolfixedpoint_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::sunnonlinsolfixedpoint_shared "${_IMPORT_PREFIX}/lib/libsundials_sunnonlinsolfixedpoint.so.3.0.0" )

# Import target "SUNDIALS::cvode_static" for configuration ""
set_property(TARGET SUNDIALS::cvode_static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::cvode_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_cvode.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::cvode_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::cvode_static "${_IMPORT_PREFIX}/lib/libsundials_cvode.a" )

# Import target "SUNDIALS::cvode_shared" for configuration ""
set_property(TARGET SUNDIALS::cvode_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SUNDIALS::cvode_shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsundials_cvode.so.6.0.0"
  IMPORTED_SONAME_NOCONFIG "libsundials_cvode.so.6"
  )

list(APPEND _IMPORT_CHECK_TARGETS SUNDIALS::cvode_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_SUNDIALS::cvode_shared "${_IMPORT_PREFIX}/lib/libsundials_cvode.so.6.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
