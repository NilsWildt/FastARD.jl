#!/usr/bin/env julia

# Local documentation build script
# Run this from the project root with: julia docs/build_docs.jl

using Pkg

println("Setting up documentation environment...")

# Activate docs environment
Pkg.activate("docs")

# Add the current package in development mode
Pkg.develop(PackageSpec(path="."))

# Install/update dependencies
Pkg.instantiate()

println("Building documentation...")

# Build documentation
include("make.jl")

println("Documentation built successfully!")
println("Open docs/build/index.html in your browser to view the documentation.")