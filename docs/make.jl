using FastARD
using Documenter

DocMeta.setdocmeta!(FastARD, :DocTestSetup, :(using FastARD); recursive = true)

# Workaround for JSON serialization issue with Julia 1.12 and JSON.jl 1.0
if VERSION >= v"1.12"
    import JSON
    # Store the original json method
    const original_json = JSON.json

    # Define a new json method that handles Dict{Symbol, Any}
    function JSON.json(d::Dict{Symbol, Any}, indent::Int)
        # Convert to a regular Dict with string keys
        str_dict = Dict{String, Any}(string(k) => v for (k, v) in d)
        # Call the original method with the converted dict
        return original_json(str_dict, indent)
    end
end

makedocs(;
    modules = [FastARD],
    authors = "Nils Wildt <nils.wildt@iws.uni-stuttgart.de>",
    repo = "https://github.com/NilsWildt/FastARD.jl/blob/{commit}{path}#{line}",
    sitename = "FastARD.jl",
    format = Documenter.HTML(;
        canonical = "https://NilsWildt.github.io/FastARD.jl",
        assets = String[],
        prettyurls = get(ENV, "CI", "false") == "true"
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "01-getting-started.md",
        "Tutorial" => "02-tutorial.md",
        "Examples" => "03-examples.md",
        "API Reference" => "95-reference.md",
    ],
    checkdocs = :exports,
    doctest = true
)

deploydocs(; repo = "github.com/NilsWildt/FastARD.jl")
