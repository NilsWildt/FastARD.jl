# Loaded automatically when both FastARD and AppleAccelerate are present in the
# environment. `using AppleAccelerate` swaps the BLAS/LAPACK backend to Apple's
# Accelerate framework as a load-time side effect; no symbols are referenced.
module FastARDAppleAccelerateExt

using AppleAccelerate

end
