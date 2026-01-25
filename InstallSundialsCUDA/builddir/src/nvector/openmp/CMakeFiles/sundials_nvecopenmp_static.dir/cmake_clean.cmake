file(REMOVE_RECURSE
  "libsundials_nvecopenmp.a"
  "libsundials_nvecopenmp.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_nvecopenmp_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
