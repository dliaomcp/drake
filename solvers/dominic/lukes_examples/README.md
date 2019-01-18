# Introduction

The [BUILD.bazel](BUILD.bazel) file has examples of how to build the following
targets:
* C library (in place square matrix transpose)
* C++ library (in place square matrix transpose)
* C program that uses the C library
* C++ program that uses the C and C++ libraries, as well as Eigen
* Googletest program that tests the C library function
* Googletest program that tests the C++ library function

These examples are completely independent of the drake C++ code but make use of
the build and test infrastructure -- this should help get you to the point where
you are writing code more quickly and not have to worry about setting up build
or test infrastructure.

# Preliminary steps
There a few things you'll need in order to build. These amount to installing
various dependencies.

## macOS
From the root directory of drake, run:

    $ ./setup/mac/install_prereqs.sh

## Ubuntu 16.04
From the root directory of drake, run:

    $ ./setup/ubuntu/16.04/install_prereqs.sh

## Ubuntu 18.04
From the root directory of drake, run:

    $ ./setup/ubuntu/18.04/install_prereqs.sh

# Building, running, and testing the examples
The folloing commands can be run from anywhere in the drake source tree. To
build just the examples above (recommended):

    $ bazel build //solvers/dominic/...

To run [example_c](example.c) or [example_cc](example.cc):

    $ bazel run //solvers/dominic:example_c
    $ bazel run //solvers/dominic:example_cc

This will rebuild the programs if any dependencies have changed.

Alternatively, these files can be run directly (i.e., without bazel), assuming
they have been built:

    $ ./bazel-bin/solvers/dominic/example_c
    $ ./bazel-bin/solvers/dominic/example_cc

To run (and rebuild if necessary) the tests:

    $ bazel test //solvers/dominic:transpose_c_test
    $ bazel test //solvers/dominic:transpose_cxx_test

Similarly, these programs may be run directly (i.e., without bazel), assuming
they have been built:

    $ ./bazel-bin/solvers/dominic/transpose_test
    $ ./bazel-bin/solvers/dominic/transpose_cxx_test

If you want to see the commands that bazel is running, add the `-s` flag to your
invocation of bazel:

    $ bazel build //solvers/dominic:example_c -s

