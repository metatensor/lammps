# metatensor requires C++17 due to Torch requiring C++17
if(CMAKE_CXX_STANDARD LESS 17)
  message(FATAL_ERROR "The ML-METATENSOR package requires the C++ standard to
be set to at least C++17")
endif()

if (BUILD_OMP AND APPLE)
    message(FATAL_ERROR
        "Can not enable both BUILD_OMP and PGK_ML-METATENSOR on Apple systems, "
        "since this results in two different versions of libiomp5.dylib (one "
        "from the system and one from Torch) being linked to the final "
        "executable, which then segfaults"
    )
endif()

# Bring the `torch` target in scope to allow evaluation
# of cmake generator expression from `metatensor_torch`
find_package(Torch REQUIRED)

# The caffe2::mkl target contains MKL_INCLUDE_DIR in it's
# INTERFACE_INCLUDE_DIRECTORIES even if MKL was not found, causing a build
# failure with "Imported target "torch" includes non-existent path" down the
# line. This code removes the missing path from INTERFACE_INCLUDE_DIRECTORIES,
# allowing the build to continue further.
if (TARGET caffe2::mkl)
    get_target_property(CAFFE2_MKL_INCLUDE_DIRECTORIES caffe2::mkl INTERFACE_INCLUDE_DIRECTORIES)
    set(MKL_INCLUDE_DIR_NOTFOUND "")
    foreach(_include_dir_ ${CAFFE2_MKL_INCLUDE_DIRECTORIES})
        if ("${_include_dir_}" MATCHES "MKL_INCLUDE_DIR-NOTFOUND")
            set(MKL_INCLUDE_DIR_NOTFOUND "${_include_dir_}")
        endif()
    endforeach()

    if (NOT "${MKL_INCLUDE_DIR_NOTFOUND}" STREQUAL "")
        list(REMOVE_ITEM CAFFE2_MKL_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR_NOTFOUND}")
    endif()
    set_target_properties(caffe2::mkl PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CAFFE2_MKL_INCLUDE_DIRECTORIES}"
    )
endif()

########### definition of metatensor and metatensor-torch targets ###########

include(FetchContent)

set(URL_BASE "https://github.com/lab-cosmo/metatensor/releases/download")

set(METATENSOR_CORE_VERSION "0.1.9")
FetchContent_Declare(metatensor
    URL ${URL_BASE}/metatensor-core-v${METATENSOR_CORE_VERSION}/metatensor-core-cxx-${METATENSOR_CORE_VERSION}.tar.gz
    URL_HASH SHA1=8c1c7ff41e0df603fcee94fdb1c2313f39ef2745
)

message(STATUS "Fetching metatensor v${METATENSOR_CORE_VERSION} from github")
FetchContent_MakeAvailable(metatensor)


set(METATENSOR_TORCH_VERSION "0.5.3")
FetchContent_Declare(metatensor-torch
    URL ${URL_BASE}/metatensor-torch-v${METATENSOR_TORCH_VERSION}/metatensor-torch-cxx-${METATENSOR_TORCH_VERSION}.tar.gz
    URL_HASH SHA1=99fee23d7ce9b9aaf4e3411034eaeba58bfe90b4
)

message(STATUS "Fetching metatensor-torch v${METATENSOR_TORCH_VERSION} from github")
FetchContent_MakeAvailable(metatensor-torch)


################ lammps target modifications ################

target_link_libraries(lammps PRIVATE metatensor_torch)
