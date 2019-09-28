#!/bin/bash
# Convert the mesh to OpenFOAM format:
gmshToFoam  main.msh
# Adjust polyMesh/boundary:
changeDictionary 
# Finally, run the simulation:
simpleFoam > log &
pid=$!
sleep 5
# Plot residuals
gnuplot residuals - &
#------------------------------------------------------------------------------
wait $pid
kill 0