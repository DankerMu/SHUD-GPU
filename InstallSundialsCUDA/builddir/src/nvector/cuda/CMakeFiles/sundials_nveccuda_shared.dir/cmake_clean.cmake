file(REMOVE_RECURSE
  "libsundials_nveccuda.pdb"
  "libsundials_nveccuda.so"
  "libsundials_nveccuda.so.6"
  "libsundials_nveccuda.so.6.0.0"
)

# Per-language clean rules from dependency scanning.
foreach(lang C CUDA)
  include(CMakeFiles/sundials_nveccuda_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
