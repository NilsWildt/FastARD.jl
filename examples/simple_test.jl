using FastARD
using Random, Statistics, LinearAlgebra
using Revise
using MultiFloats


function run(noise_std = 0.1)
    Random.seed!(1)

    # Create a sparse regression problem
    n_samples, n_features = 200, 100
    X_raw = randn(n_samples, n_features)

    # No need to manually standardize - FastARD now does this automatically by default (standardize=true)
    X = X_raw
    local true_coef
    # True sparse coefficient vector
    true_coef = zeros(n_features)
    active_features = [5, 12, 23, 45, 67, 89]  # 6 out of 100 features are active
    true_coef[active_features] = [2.5, -1.8, 3.2, -0.9, 1.6, -2.1]

    # Generate observations
    # noise_std = 10.0
    y = X * true_coef + noise_std * randn(n_samples)

    println("True number of active features: ", length(active_features))
    println("Signal-to-noise ratio: ", std(X * true_coef) / noise_std)


    # Create and fit the model with new default parameters
    model = FastARDRegressor(verbose = true, compute_score = true)
    fit!(model, X, y)

    # Analyze the results
    active_indices, active_coefs = get_active_coefficients(model)
    println("\nSelected features: ", active_indices)
    println("True active features: ", active_features)

    # Check feature selection accuracy
    correctly_selected = length(intersect(active_indices, active_features))
    precision = correctly_selected / length(active_indices)
    recall = correctly_selected / length(active_features)

    println("Precision: $precision")
    println("Recall: $recall")

    # Additional analysis
    println("Total selected: $(length(active_indices))")
    println("Signal-to-noise ratio: $(std(X * true_coef) / noise_std)")
    println("Effective degrees of freedom: $(sum(model.active))")
    println("Final noise precision (β): $(model.beta)")

    # Check if the model is overfitting
    training_rmse = sqrt(mean((FastARD.predict(model, X) .- y) .^ 2))
    println("Training RMSE: $training_rmse")


    # using Makie
    # using CairoMakie

    # if !isempty(model.scores)
    #     fig = Figure(size = (800, 600))
    #     ax = Axis(fig[1, 1],
    #               xlabel = "Iteration",
    #               ylabel = "Log Marginal Likelihood",
    #               title = "Convergence")
    #     lines!(ax, 1:length(model.scores), model.scores, linewidth = 2)
    #     display(fig)
    # end


    # High-dimensional problem
    n_samples, n_features = 50, 500  # More features than samples
    X_raw = randn(n_samples, n_features)
    # Standardize features
    X = (X_raw .- mean(X_raw, dims = 1)) ./ std(X_raw, dims = 1)

    # Very sparse solution
    true_coef = zeros(n_features)
    true_coef[1:3] = [5.0, -3.0, 4.0]  # Only 3 active features
    y = X * true_coef + 0.1 * randn(n_samples)

    # Fit the model for high-dimensional case (need stronger regularization)
    model = FastARDRegressor(
        n_iter = 300, verbose = true,
        lambda_reg = 1.0e-5,   # Stronger regularization for n<p case
        max_alpha = 1.0e3
    )     # Moderate max precision
    fit!(model, X, y)

    # Check results
    active_indices, active_coefs = get_active_coefficients(model)
    println("Selected $(length(active_indices)) features out of $n_features")
    println("Active features: ", active_indices)
    println("True active features: [1, 2, 3]")

    # Prediction accuracy
    y_pred = FastARD.predict(model, X)
    println("RMSE: ", sqrt(mean((y_pred .- y) .^ 2)))


    # Generate test data
    X_test = randn(50, n_features)
    y_test_true = X_test * true_coef

    # Get predictions with uncertainty
    y_pred, y_std = predict_with_uncertainty(model, X_test)

    # Analyze uncertainty quality
    residuals = abs.(y_pred .- y_test_true)
    println("Mean prediction uncertainty: ", mean(y_std))
    println("Mean absolute error: ", mean(residuals))

    # Check if uncertainties are well-calibrated
    # Points within 1σ should contain ~68% of true values
    within_1sigma = sum(residuals .<= y_std) / length(residuals)
    within_2sigma = sum(residuals .<= 2 .* y_std) / length(residuals)
    println("Fraction within 1σ: $within_1sigma (should be ~0.68)")
    println("Fraction within 2σ: $within_2sigma (should be ~0.95)")

    # Uncertainty diagnostics
    println("Uncertainty range: $(minimum(y_std)) to $(maximum(y_std))")
    println("Residual range: $(minimum(residuals)) to $(maximum(residuals))")
    println("Ratio mean(residuals)/mean(std): $(mean(residuals) / mean(y_std))")


    # Create multicollinear features
    n_samples, n_base_features = 100, 20
    X_base = randn(n_samples, n_base_features)

    # Add correlated copies of some features
    X_corr1 = X_base[:, 1:5] + 0.1 * randn(n_samples, 5)  # Noisy copies
    X_corr2 = 2.0 * X_base[:, 1:5]  # Scaled copies

    X = [X_base X_corr1 X_corr2]  # 20 + 5 + 5 = 30 features
    n_features = size(X, 2)

    # True coefficients only on original features
    true_coef = zeros(n_features)
    true_coef[1:5] = [1.0, -1.5, 2.0, -0.5, 1.8]
    y = X * true_coef + 0.1 * randn(n_samples)

    # Fit model
    model = FastARDRegressor(verbose = false)
    fit!(model, X, y)

    active_indices, active_coefs = get_active_coefficients(model)
    println("Selected features: ", active_indices)
    println("Features 1-5: original")
    println("Features 21-25: noisy copies of 1-5")
    println("Features 26-30: scaled copies of 1-5")


    # Multiple output problem
    n_outputs = 3
    Y = randn(n_samples, n_outputs)

    # Each output has different sparsity pattern
    for i in 1:n_outputs
        local true_coef = zeros(n_features)
        local active_features = rand(1:n_features, 3 + i)  # Different number of active features
        true_coef[active_features] = randn(length(active_features))
        Y[:, i] = X * true_coef + 0.1 * randn(n_samples)
    end

    # Fit models (returns vector of models)
    models = fit!(deepcopy(model), X, Y)

    # Analyze each output
    for i in 1:n_outputs
        local active_indices, _ = get_active_coefficients(models[i])
        println("Output $i: $(length(active_indices)) active features")
    end


    # High precision computation
    model_hp = FastARDRegressor(MultiFloat{Float64, 2}, verbose = true)
    fit!(model_hp, MultiFloat{Float64, 2}.(X), MultiFloat{Float64, 2}.(y[:, 1]))

    # Single precision for speed
    model_sp = FastARDRegressor(Float32, verbose = true)
    fit!(model_sp, Float32.(X), Float32.(y[:, 1]))

    # Compare results
    active_hp, coef_hp = get_active_coefficients(model_hp)
    active_sp, coef_sp = get_active_coefficients(model_sp)

    println("High precision active features: ", active_hp)
    return println("Single precision active features: ", active_sp)
end

run()
