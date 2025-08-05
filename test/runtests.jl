using FastARD
using Test

@testset "FastARD.jl" begin
    @test FastARD.hello_world() == "Hello, World!"
end
