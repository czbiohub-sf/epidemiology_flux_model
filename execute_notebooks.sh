#!/bin/bash
# This script execute all notebooks. It must be run from within the docker container.

for fname in 1-clustering \
             2-fluxes \
             3-import_cssegi \
             4-distances \
             5-SIR_dynamics_fit \
             6-simulations/61-simplified_model \
             6-simulations/62-distance_cutoff \
             6-simulations/63-lattice_2d_nneighbors_loop_beta \
             6-simulations/64-lattice_2d_nneighbors_loop_n2 \
             7-wave_analysis/71-ode_gamma_eq0 \
             7-wave_analysis/72-ode_gamma_neq0 \
             8-infectivity_matrix_properties
do
  echo Executing notebook ${fname}.ipynb
  jupyter nbconvert --to notebook --inplace --execute notebooks/${fname}.ipynb
done

