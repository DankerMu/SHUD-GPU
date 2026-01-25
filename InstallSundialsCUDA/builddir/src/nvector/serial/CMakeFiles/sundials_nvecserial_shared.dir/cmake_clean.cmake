file(REMOVE_RECURSE
  "libsundials_nvecserial.pdb"
  "libsundials_nvecserial.so"
  "libsundials_nvecserial.so.6"
  "libsundials_nvecserial.so.6.0.0"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/sundials_nvecserial_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
