The base package can be compiled as
cmake ../cmake -DPKG_ML-METATENSOR=ON -DCMAKE_PREFIX_PATH=/.../site-packages/torch/share/cmake/
where /.../site-packages/torch/ is the path to a pip installation of torch

The kokkos version should be compiled as
cmake ../cmake/ -DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON -DPKG_ML-METATENSOR=ON -DCMAKE_PREFIX_PATH=/.../libtorch/share/cmake/
where /.../libtorch/ is the path to a libtorch C++11 ABI distribution (which can be downloaded from https://pytorch.org/get-started/locally/).
The OpenMP version (as opposed to the CUDA version) can be enabled with -DKokkos_ENABLE_OPENMP=ON instead of -DKokkos_ENABLE_CUDA=ON

The consistency between the two interfaces can be checked with
../../../build/lmp -k on g 1 -pk kokkos newton on -in in.kokkos.metatensor
(or `t Nt` instead of `g 1` for an OpenMP run with Nt threads) 
and the output can be compared with that of the plain metatensor interface
../../../build/lmp -in in.metatensor
