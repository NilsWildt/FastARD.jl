# FastARD Examples

This directory contains comprehensive examples demonstrating various applications and use cases of FastARD.jl. Each file is a complete, runnable example that showcases different aspects of the library.

## Available Examples

### 1. `simple_test.jl` (Original)
**Basic functionality demonstration**
- Basic sparse regression
- High-dimensional regression (n < p)
- Uncertainty quantification
- Multicollinearity handling
- Multiple output regression
- Different precision types

### 2. `polynomial_feature_test.jl`
**Automatic polynomial feature selection**
- Nonlinear regression with polynomial terms
- Feature selection for polynomial degrees 0-10
- Extrapolation performance analysis
- Uncertainty calibration assessment
- Visualization of results and convergence

**Key concepts:** Nonlinear modeling, feature engineering, overfitting prevention

### 3. `image_denoising_test.jl`
**Sparse coding for image processing**
- Sparse dictionary representation of image patches
- Denoising through sparse reconstruction
- Dictionary atom usage analysis
- Robustness to different noise levels
- Performance and computational analysis

**Key concepts:** Sparse coding, overcomplete dictionaries, image processing

### 4. `genomics_test.jl`
**High-dimensional genomics data analysis**
- Gene expression feature selection
- Pathway correlation structure simulation
- Cross-validation for model validation
- Feature importance ranking
- Biomarker discovery simulation

**Key concepts:** High-dimensional data, correlated features, biological data analysis

### 5. `timeseries_test.jl`
**Time series feature engineering and forecasting**
- Automatic lag selection
- Seasonal pattern detection
- Trend and structural break identification
- Multi-step ahead forecasting
- Rolling window validation
- Feature stability analysis

**Key concepts:** Time series analysis, temporal dependencies, forecasting

### 6. `signal_recovery_test.jl`
**Compressed sensing and sparse signal reconstruction**
- Multiple compression scenarios
- Different measurement matrix types (Gaussian, Bernoulli, Partial Fourier)
- Phase transition analysis
- Coherence impact assessment
- Structured sparsity recovery

**Key concepts:** Compressed sensing, signal processing, measurement matrices

### 7. `parameter_tuning_test.jl`
**Comprehensive parameter optimization guide**
- Convergence parameter effects
- Numerical stability analysis
- Problem-specific configurations
- Performance vs accuracy trade-offs
- Cross-validation for parameter selection
- Configuration recommendations

**Key concepts:** Hyperparameter tuning, numerical stability, performance optimization

## Usage

Each example can be run independently:

```julia
# Run a specific example
julia examples/polynomial_feature_test.jl

# Or from Julia REPL
include("examples/genomics_test.jl")
```

## Example Complexity Levels

**Beginner:** Start with `simple_test.jl` and `polynomial_feature_test.jl`
**Intermediate:** Try `genomics_test.jl` and `timeseries_test.jl`  
**Advanced:** Explore `signal_recovery_test.jl` and `parameter_tuning_test.jl`
**Expert:** `image_denoising_test.jl` for specialized applications

## Common Patterns Across Examples

### Data Preparation
- Feature standardization
- Correlation structure creation
- Noise addition and SNR control

### Model Evaluation
- Precision, recall, and F1-score for feature selection
- RMSE and R² for prediction accuracy
- Uncertainty calibration assessment

### Robustness Testing
- Cross-validation
- Bootstrap analysis
- Parameter sensitivity analysis
- Multiple noise levels

### Performance Analysis
- Computational timing
- Memory usage estimation
- Scalability assessment

## Dependencies

Most examples require only:
```julia
using FastARD
using Random, Statistics, LinearAlgebra
```

Some examples have additional optional dependencies:
- `CairoMakie.jl` for visualization (polynomial_feature_test.jl)
- `MultiFloats.jl` for high-precision arithmetic (several examples)

## Tips for Running Examples

1. **Set Random Seed**: All examples use `Random.seed!()` for reproducibility
2. **Check Output**: Each example produces detailed console output explaining results
3. **Timing**: Some examples (signal recovery, genomics) may take several minutes
4. **Memory**: Large examples (genomics, signal recovery) use substantial memory
5. **Customization**: All parameters can be easily modified for different problem sizes

## Learning Path

1. **Start with `simple_test.jl`** to understand basic FastARD usage
2. **Choose domain-specific examples** based on your application area
3. **Study `parameter_tuning_test.jl`** to optimize performance for your problems
4. **Combine concepts** from multiple examples for complex applications

## Contributing

When adding new examples:
- Follow the established pattern with clear section headers
- Include comprehensive output and analysis
- Add timing and performance metrics
- Document any special dependencies
- Update this README with the new example description