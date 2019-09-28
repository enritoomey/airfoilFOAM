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
pid2=$!
#------------------------------------------------------------------------------
wait $pid
kill $pid2