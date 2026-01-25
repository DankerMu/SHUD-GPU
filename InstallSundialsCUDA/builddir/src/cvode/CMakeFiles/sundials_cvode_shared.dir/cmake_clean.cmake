file(REMOVE_RECURSE
  "libsundials_cvode.pdb"
  "libsundials_cvode.so"
  "libsundials_cvode.so.6"
  "libsundials_cvode.so.6.0.0"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_cvode_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
