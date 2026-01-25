file(REMOVE_RECURSE
  "libsundials_sunlinsolspbcgs.pdb"
  "libsundials_sunlinsolspbcgs.so"
  "libsundials_sunlinsolspbcgs.so.4.0.0"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_sunlinsolspbcgs_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
