file(REMOVE_RECURSE
  "libsundials_nvecopenmp.pdb"
  "libsundials_nvecopenmp.so"
  "libsundials_nvecopenmp.so.6"
  "libsundials_nvecopenmp.so.6.0.0"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_nvecopenmp_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
