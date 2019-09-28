# airfoilFOAM

Python script to run CFD analysis on airfoil using **OpenFOAM** to simulate and **gmsh** to generate mesh.
Given a **.dat** file with the geometry of the airfoil, and a set of angles of attack (AoA), the script generates the
mesh for the geometry and runs a simulation for each AoA, returning the **ClvsAlpha**, **CdvsCl** and **CmvsAlpha**
 curves.
 
 ## Requirements
 This scripts requires python 3.5> installed, **gmesh** and **openFoam**. OpenFoam may be run native, or
 thought docker. In case you don't have any, run script **openfoam7-linux**. This script will download a docker
 image with openfoam and paraview installed, and mount a container in background, named **openfoam**. In order for
 this to work, docker must be installed and your user added to the docker group. See more instructions
 in https://openfoam.org/download/7-linux/.
 
 We strongly recomend to work with a python virtual environment. For example:
```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
``` 

