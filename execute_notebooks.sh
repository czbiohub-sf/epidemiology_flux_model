#!/bin/bash
# This script execute all notebooks. It must be run from within the docker container.

for fname in 1-clustering 2-fluxes 3-import_cssegi 4-distances 5-SIR_dynamics_fit
do
  echo Executing notebook ${fname}.ipynb
  jupyter nbconvert --to notebook --inplace --execute notebooks/${fname}.ipynb
done

