using TestItemRunner

# Discover and run every @testitem in the package (test/ and src/). Each testitem
# runs in its own module and declares its own imports, so tests are isolated and
# can be run individually from an editor.
@run_package_tests
