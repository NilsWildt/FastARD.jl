{
  description = "FastARD - A Julia package for fast automatic relevance determination";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    claude-code.url = "github:sadjow/claude-code-nix";
  };

  outputs = { self, nixpkgs, flake-utils, claude-code }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Create development environment for the main project
        devEnv = pkgs.mkShell {
          name = "fastard-dev";
          
          buildInputs = with pkgs; [
            # Additional development tools
            git
            gcc
            gfortran
            # For CairoMakie plotting
            cairo
            pango
            gdk-pixbuf
          ] ++ [
            # Claude Code from external flake
            claude-code.packages.${system}.default
          ];
          
          shellHook = ''
            echo "FastARD development environment loaded"
            echo "Using global Julia installation"
            echo ""
            echo "Available commands:"
            echo "  julia --project=. -e 'using Pkg; Pkg.instantiate()'  # Install dependencies"
            echo "  julia --project=. -e 'using FastARD'                  # Test import"
            echo "  julia --project=. test/runtests.jl                    # Run tests"
            echo "  julia --project=docs docs/build_docs.jl               # Build documentation"
            echo "  claude-code                                           # Start Claude Code"
          '';
          
          # Environment variables for Julia
          JULIA_DEPOT_PATH = "~/.julia";
          JULIA_NUM_THREADS = "auto";
          
          # For CairoMakie
          GDK_PIXBUF_MODULE_FILE = "${pkgs.librsvg.out}/lib/gdk-pixbuf-2.0/2.10.0/loaders.cache";
        };
        
        # Documentation environment
        docsEnv = pkgs.mkShell {
          name = "fastard-docs";
          
          buildInputs = with pkgs; [
            # Documentation dependencies
            cairo
            pango
            gdk-pixbuf
          ];
          
          shellHook = ''
            echo "FastARD documentation environment loaded"
            echo ""
            echo "To build documentation:"
            echo "  julia --project=docs docs/build_docs.jl"
            echo "  julia --project=docs -e 'using LiveServer; serve(dir=\"docs/build\")'"
          '';
          
          JULIA_DEPOT_PATH = "~/.julia";
        };
        
        # Test environment
        testEnv = pkgs.mkShell {
          name = "fastard-test";
          
          buildInputs = with pkgs; [
            # Test dependencies
            gcc
            gfortran
          ];
          
          shellHook = ''
            echo "FastARD test environment loaded"
            echo ""
            echo "To run tests:"
            echo "  julia --project=test test/runtests.jl"
          '';
          
          JULIA_DEPOT_PATH = "~/.julia";
        };
        
      in {
        # Default development environment
        devShells.default = devEnv;
        
        # Specific environments
        devShells.docs = docsEnv;
        devShells.test = testEnv;
        
        # Packages
        packages = {
          # You can add package builds here if needed
        };
        
        # Apps
        apps = {
          # Development commands
          dev = {
            type = "app";
            program = toString (pkgs.writeShellScript "dev" ''
              echo "Starting FastARD development environment..."
              exec ${pkgs.bash}/bin/bash
            '');
          };
          
          # Test runner
          test = {
            type = "app";
            program = toString (pkgs.writeShellScript "test" ''
              echo "Running FastARD tests..."
              julia --project=test test/runtests.jl
            '');
          };
          
          # Documentation builder
          docs = {
            type = "app";
            program = toString (pkgs.writeShellScript "docs" ''
              echo "Building FastARD documentation..."
              julia --project=docs docs/build_docs.jl
            '');
          };
          
          # Install dependencies
          install = {
            type = "app";
            program = toString (pkgs.writeShellScript "install" ''
              echo "Installing FastARD dependencies..."
              julia --project=. -e 'using Pkg; Pkg.instantiate()'
              julia --project=docs -e 'using Pkg; Pkg.instantiate()'
              julia --project=test -e 'using Pkg; Pkg.instantiate()'
            '');
          };
        };
        
        # Formatter
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}