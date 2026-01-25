file(REMOVE_RECURSE
  "libsundials_sunlinsolspfgmr.pdb"
  "libsundials_sunlinsolspfgmr.so"
  "libsundials_sunlinsolspfgmr.so.4.0.0"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_sunlinsolspfgmr_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
