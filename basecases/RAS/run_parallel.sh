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
wait $pid
kill 0

# Recompose results
reconstructPar

#------------------------------------------------------------------------------
