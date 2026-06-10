# Loaded automatically when both FastARD and MKL are present in the environment.
# `using MKL` swaps the BLAS/LAPACK backend to Intel MKL as a load-time side
# effect; no symbols are referenced.
module FastARDMKLExt

using MKL

end
