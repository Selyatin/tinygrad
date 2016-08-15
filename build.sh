#!/usr/bin/env bash

# A simple script to build tinygrad from scratch.

# Ad-hoc approach to test if the script is executed in the root folder of tinygrad.
if [ ! -f "build.sh" ]; then
    echo "Please execute the script in the root of the tinygrad repository folder."
    exit 1
fi

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Attempt to create the makefiles for the project.
cmake "Unix makefiles" ..

if [ $? -nq 0 ]; then
    echo "[tinygrad] CMake failed. Have you installed Eigen system-wide?"
fi

# Make the project using the previously generated makefiles.
make

if [ $? -eq 0 ]; then
    echo "[tinygrad] Compilation successful. Test applications are available in folder 'build'."
else
    echo "[tinygrad] The compilation failed."
fi
