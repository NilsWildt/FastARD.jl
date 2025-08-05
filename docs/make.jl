using FastARD
using Documenter

DocMeta.setdocmeta!(FastARD, :DocTestSetup, :(using FastARD); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [FastARD],
    authors = "Nils Wildt <nils.wildt@iws.uni-stuttgart.de>",
    repo = "https://github.com/NilsWildt/FastARD.jl/blob/{commit}{path}#{line}",
    sitename = "FastARD.jl",
    format = Documenter.HTML(; canonical = "https://NilsWildt.github.io/FastARD.jl"),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/NilsWildt/FastARD.jl")
