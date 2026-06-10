#!/usr/bin/env bash
# Format every tracked Julia file in place with Runic, matching the Runic CI
# check (which runs over all tracked `*.jl`). Run this before pushing.
set -euo pipefail
git ls-files -z -- '*.jl' \
    | xargs -0 julia --project=@runic -e 'using Runic; exit(Runic.main(ARGS))' -- "--inplace"
