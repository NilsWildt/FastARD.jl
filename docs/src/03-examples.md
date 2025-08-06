# Examples

This page showcases practical applications of FastARD.jl across different domains and problem types.

## Example 1: Polynomial Feature Selection

Automatic selection of relevant polynomial terms for nonlinear regression.

```julia
using FastARD
using Random, Statistics

Random.seed!(42)

# Generate 1D data with polynomial relationship
n_samples = 100
x = sort(2 * rand(n_samples) .- 1)  # Random points in [-1, 1]

# True function: y = 2 + 3x - 1.5x² + 0.8x³ + noise
y_true = 2 .+ 3 .* x .- 1.5 .* x.^2 .+ 0.8 .* x.^3
y = y_true + 0.2 * randn(n_samples)

# Create polynomial features up to degree 10
max_degree = 10
X = zeros(n_samples, max_degree + 1)
for i in 0:max_degree
    X[:, i+1] = x.^i
end

println("Created polynomial features of degree 0 to $max_degree")

# Fit FastARD model
model = FastARDRegressor(verbose=false)
fit!(model, X, y)

# Analyze results
active_indices, active_coefs = get_active_coefficients(model)
println("Selected polynomial terms:")
for i in 1:length(active_indices)
    degree = active_indices[i] - 1
    coef = active_coefs[i]
    println("  x^$degree: coefficient = $coef")
end

# Prediction accuracy
y_pred = predict(model, X)
rmse = sqrt(mean((y_pred .- y).^2))
println("RMSE: $rmse")

# Compare with true coefficients
println("\nTrue coefficients:")
println("  x^0 (constant): 2.0")
println("  x^1 (linear): 3.0") 
println("  x^2 (quadratic): -1.5")
println("  x^3 (cubic): 0.8")
```

## Example 2: Image Denoising with Sparse Dictionaries

Using ARD for sparse coding in image denoising applications.

```julia
using FastARD
using Random

Random.seed!(123)

# Simulate image patches (8x8 patches flattened to 64-dimensional vectors)
patch_size = 64
n_patches = 500
n_atoms = 256  # Overcomplete dictionary

# Generate synthetic image patches (sparse in DCT domain)
true_sparse_codes = zeros(n_patches, n_atoms)
for i in 1:n_patches
    # Each patch uses only a few dictionary atoms
    n_active = rand(3:8)
    active_atoms = randperm(n_atoms)[1:n_active]
    true_sparse_codes[i, active_atoms] = randn(n_active)
end

# Create random dictionary
dictionary = randn(patch_size, n_atoms)
dictionary ./= sqrt.(sum(dictionary.^2, dims=1))  # Normalize columns

# Generate noisy observations
clean_patches = true_sparse_codes * dictionary'
noise_level = 0.1
noisy_patches = clean_patches + noise_level * randn(size(clean_patches))

println("Dictionary size: $(size(dictionary))")
println("Number of patches: $n_patches")
println("Average sparsity: $(mean(sum(abs.(true_sparse_codes) .> 1e-6, dims=2)))")

# Sparse coding with FastARD (process each patch)
reconstructed_patches = zeros(size(noisy_patches))
sparsity_levels = Float64[]

for i in 1:min(50, n_patches)  # Process first 50 patches for demo
    # Solve: noisy_patch[i, :] ≈ dictionary * sparse_code
    model = FastARDRegressor(verbose=false, n_iter=100)
    fit!(model, dictionary, noisy_patches[i, :])
    
    # Reconstruct patch
    reconstructed_patches[i, :] = predict(model, dictionary)
    
    # Track sparsity
    active_indices, _ = get_active_coefficients(model)
    push!(sparsity_levels, length(active_indices))
    
    if i % 10 == 0
        println("Processed $i patches...")
    end
end

# Analyze results
processed_patches = min(50, n_patches)
avg_sparsity = mean(sparsity_levels)
denoising_improvement = mean(sum((noisy_patches[1:processed_patches, :] .- clean_patches[1:processed_patches, :]).^2, dims=2)) -
                       mean(sum((reconstructed_patches[1:processed_patches, :] .- clean_patches[1:processed_patches, :]).^2, dims=2))

println("\nDenoising Results:")
println("Average sparsity: $avg_sparsity atoms per patch")
println("Denoising improvement (MSE reduction): $denoising_improvement")
```

## Example 3: Gene Expression Analysis

Feature selection in high-dimensional genomics data.

```julia
using FastARD
using Random

Random.seed!(456)

# Simulate gene expression data
n_samples = 200      # Patients
n_genes = 5000       # Genes (features)
n_relevant = 25      # Relevant genes

# Generate correlated gene expression data
X = randn(n_samples, n_genes)

# Add correlation structure (genes in pathways are correlated)
pathway_size = 50
n_pathways = n_genes ÷ pathway_size
for p in 1:n_pathways
    pathway_genes = ((p-1)*pathway_size + 1):(p*pathway_size)
    # Add common pathway effect
    pathway_effect = randn(n_samples)
    X[:, pathway_genes] .+= 0.5 * pathway_effect
end

# Create disease phenotype from subset of genes
relevant_genes = randperm(n_genes)[1:n_relevant]
true_effects = randn(n_relevant)
disease_score = X[:, relevant_genes] * true_effects

# Add noise and convert to binary outcome
noise = randn(n_samples)
y_continuous = disease_score + 0.5 * noise
y = Float64.(y_continuous .> median(y_continuous))  # Binary classification target

println("Simulated genomics data:")
println("  Samples: $n_samples")
println("  Genes: $n_genes") 
println("  Relevant genes: $n_relevant")
println("  Disease prevalence: $(mean(y))")

# Apply FastARD for gene selection
model = FastARDRegressor(verbose=false, n_iter=300)
fit!(model, X, y_continuous)  # Use continuous target for regression

# Analyze gene selection
selected_genes, gene_effects = get_active_coefficients(model)
n_selected = length(selected_genes)

# Check overlap with true relevant genes
correctly_identified = length(intersect(selected_genes, relevant_genes))
precision = correctly_identified / n_selected
recall = correctly_identified / n_relevant

println("\nGene Selection Results:")
println("  Selected genes: $n_selected")
println("  Correctly identified: $correctly_identified")
println("  Precision: $precision")
println("  Recall: $recall")

# Top selected genes by effect size
sorted_idx = sortperm(abs.(gene_effects), rev=true)
println("\nTop 10 selected genes by effect size:")
for i in 1:min(10, length(sorted_idx))
    gene_idx = selected_genes[sorted_idx[i]]
    effect = gene_effects[sorted_idx[i]]
    is_relevant = gene_idx in relevant_genes ? "✓" : "✗"
    println("  Gene $gene_idx: effect = $effect $is_relevant")
end
```

## Example 4: Time Series Feature Engineering

Automatic selection of relevant lags and transformations in time series prediction.

```julia
using FastARD
using Random

Random.seed!(789)

# Generate synthetic time series
n_points = 500
t = 1:n_points

# True time series with trend, seasonality, and AR components
trend = 0.01 * t
seasonal = 2 * sin.(2π * t / 50) + cos.(2π * t / 25)
ar_component = zeros(n_points)

# AR(2) process: y[t] = 0.7*y[t-1] - 0.2*y[t-2] + noise
for i in 3:n_points
    ar_component[i] = 0.7 * ar_component[i-1] - 0.2 * ar_component[i-2] + 0.5 * randn()
end

y = trend + seasonal + ar_component + 0.2 * randn(n_points)

println("Generated time series with $n_points points")

# Create lagged features
max_lag = 20
feature_names = String[]
X = zeros(n_points - max_lag, 0)

# Add lagged values
for lag in 1:max_lag
    lagged_y = y[(max_lag-lag+1):(n_points-lag)]
    X = [X lagged_y]
    push!(feature_names, "lag_$lag")
end

# Add trend features
time_features = collect((max_lag+1):n_points)
X = [X time_features time_features.^2]
push!(feature_names, "trend_linear", "trend_quadratic")

# Add seasonal features
for period in [10, 25, 50]
    sin_feature = sin.(2π * time_features / period)
    cos_feature = cos.(2π * time_features / period)
    X = [X sin_feature cos_feature]
    push!(feature_names, "sin_$period", "cos_$period")
end

# Target (current values)
y_target = y[(max_lag+1):end]

println("Created $(size(X, 2)) features:")
println("  Lags: $max_lag")
println("  Trend: 2 features")
println("  Seasonal: $(2*3) features")

# Fit FastARD model
model = FastARDRegressor(verbose=false)
fit!(model, X, y_target)

# Analyze selected features
selected_indices, selected_coefs = get_active_coefficients(model)
println("\nSelected features:")
for i in 1:length(selected_indices)
    feat_idx = selected_indices[i]
    feat_name = feature_names[feat_idx]
    coef = selected_coefs[i]
    println("  $feat_name: coefficient = $coef")
end

# Prediction accuracy
y_pred = predict(model, X)
rmse = sqrt(mean((y_pred .- y_target).^2))
mae = mean(abs.(y_pred .- y_target))

println("\nPrediction Performance:")
println("  RMSE: $rmse")
println("  MAE: $mae")

# One-step-ahead prediction on test set
test_size = 50
train_X = X[1:(end-test_size), :]
train_y = y_target[1:(end-test_size)]
test_X = X[(end-test_size+1):end, :]
test_y = y_target[(end-test_size+1):end]

model_test = FastARDRegressor(verbose=false)
fit!(model_test, train_X, train_y)

test_pred, test_std = predict_with_uncertainty(model_test, test_X)
test_rmse = sqrt(mean((test_pred .- test_y).^2))

println("  Test RMSE: $test_rmse")
println("  Mean prediction uncertainty: $(mean(test_std))")
```

## Example 5: Sparse Signal Recovery

Compressed sensing and sparse signal reconstruction.

```julia
using FastARD
using Random

Random.seed!(101)

# Sparse signal recovery problem
signal_length = 1000
n_measurements = 300  # Undersampled (n_measurements < signal_length)
sparsity_level = 50   # Number of non-zero elements

# Generate sparse signal
true_signal = zeros(signal_length)
sparse_support = randperm(signal_length)[1:sparsity_level]
true_signal[sparse_support] = randn(sparsity_level)

println("Signal recovery problem:")
println("  Signal length: $signal_length")
println("  Measurements: $n_measurements")
println("  True sparsity: $sparsity_level")
println("  Compression ratio: $(n_measurements/signal_length)")

# Measurement matrix (random Gaussian)
measurement_matrix = randn(n_measurements, signal_length)
measurement_matrix ./= sqrt.(sum(measurement_matrix.^2, dims=1))  # Normalize

# Noisy measurements
noise_level = 0.01
measurements = measurement_matrix * true_signal + noise_level * randn(n_measurements)

# Recover signal using FastARD
model = FastARDRegressor(verbose=false, n_iter=400)
fit!(model, measurement_matrix, measurements)

# Analyze recovery
recovered_indices, recovered_coefs = get_active_coefficients(model)
recovered_signal = zeros(signal_length)
recovered_signal[recovered_indices] = recovered_coefs

# Performance metrics
correctly_recovered = length(intersect(recovered_indices, sparse_support))
false_positives = length(setdiff(recovered_indices, sparse_support))
missed = length(setdiff(sparse_support, recovered_indices))

precision = correctly_recovered / length(recovered_indices)
recall = correctly_recovered / sparsity_level
f1_score = 2 * precision * recall / (precision + recall)

signal_error = norm(recovered_signal - true_signal) / norm(true_signal)

println("\nRecovery Results:")
println("  Recovered elements: $(length(recovered_indices))")
println("  Correctly recovered: $correctly_recovered")
println("  False positives: $false_positives")
println("  Missed elements: $missed")
println("  Precision: $precision")
println("  Recall: $recall")
println("  F1-score: $f1_score")
println("  Relative signal error: $signal_error")

# Check measurement fidelity
measurement_error = norm(measurement_matrix * recovered_signal - measurements) / norm(measurements)
println("  Measurement error: $measurement_error")
```

## Performance Tips

### For Large-Scale Problems

```julia
# Use single precision for speed
model = FastARDRegressor(Float32, n_iter=100, verbose=false)

# Process data in chunks for memory efficiency
chunk_size = 1000
n_chunks = ceil(Int, size(X, 1) / chunk_size)

for chunk in 1:n_chunks
    start_idx = (chunk - 1) * chunk_size + 1
    end_idx = min(chunk * chunk_size, size(X, 1))
    
    X_chunk = X[start_idx:end_idx, :]
    y_chunk = y[start_idx:end_idx]
    
    # Process chunk...
end
```

### For High-Precision Requirements

```julia
using MultiFloats

# Use arbitrary precision arithmetic
model = FastARDRegressor(MultiFloat{Float64,4}, tol=1e-12)
```

### For Real-Time Applications

```julia
# Pre-allocate and reuse models
model = FastARDRegressor(n_iter=50, verbose=false)

# Warm-up compilation
dummy_X = randn(10, 5)
dummy_y = randn(10)
fit!(model, dummy_X, dummy_y)

# Now use for real-time processing...
```

These examples demonstrate FastARD.jl's versatility across different domains. The key is to:

1. **Understand your sparsity structure** - ARD works best when true sparsity exists
2. **Engineer relevant features** - Include candidate features that might be relevant
3. **Monitor convergence** - Check that the algorithm has converged properly
4. **Validate results** - Use cross-validation or held-out data to assess performance
5. **Interpret uncertainty** - Use prediction uncertainties for decision making