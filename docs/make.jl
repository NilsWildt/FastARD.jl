using FastARD
using Documenter

DocMeta.setdocmeta!(FastARD, :DocTestSetup, :(using FastARD); recursive = true)

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
        "API Reference" => "95-reference.md"
    ],
    checkdocs = :exports,
    doctest = true
)

deploydocs(; repo = "github.com/NilsWildt/FastARD.jl")
