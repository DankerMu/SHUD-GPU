file(REMOVE_RECURSE
  "libsundials_nveccuda.a"
  "libsundials_nveccuda.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C CUDA)
  include(CMakeFiles/sundials_nveccuda_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
