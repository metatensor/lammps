Design taken from pace_kokkos with accessory files in their own directory.
Will probably need some cmake magic to copy them here from somewhere else.

To be compiled as
cmake ../cmake/ -DPKG_ML-METATENSOR=ON -DPKG_KOKKOS=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON

Run the example with
../../../build/lmp -k on g 1 -pk kokkos newton on -in in.metatensor_kokkos
and compare its output with the non-kokkos interface
../../../build/lmp -in in.metatensor


cmake ../cmake -DPKG_ML-METATENSOR=ON -DCMAKE_PREFIX_PATH=/home/filippo/code/virtualenvs/base/lib/python3.12/site-packages/torch/share/cmake/
cmake ../cmake/ -DPKG_ML-METATENSOR=ON -DPKG_KOKKOS=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DCMAKE_PREFIX_PATH=/home/filippo/code/virtualenvs/base/lib/python3.12/site-packages/torch/share/cmake/
