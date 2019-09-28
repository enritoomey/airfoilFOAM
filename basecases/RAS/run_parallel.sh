#!/bin/bash
# Convert the mesh to OpenFOAM format:
gmshToFoam  main.msh
# Adjust polyMesh/boundary:
changeDictionary
# Decompose
decomposePar
# Run the simulation in parallel:
mpirun -np $1 simpleFoam -parallel > log &
pid=$!
sleep 5
# Plot residuals
gnuplot residuals - &
pid2=$!

wait $pid
kill $pid2

# Recompose results
reconstructPar

#------------------------------------------------------------------------------
