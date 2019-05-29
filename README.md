# Recurrent Kalman Networks

This is a new, cleaned up version of the Code used for the experiments described in the paper. We highly recommend you use this version, however, you can find the old version in the branch "icml_submission". 


## Code

### n_link_sim
  Contains simulator for quad_link, see seperate readme
  
### rkn
#### data
Code to generate the needed data sets

### rkn
Implementation of the RKN as described in the paper
  - RKN: The full RKN consisting of encoder, RKNTransitionCell and decoder, implemented as subclass of keras.models.Model. Still abstract, the hidden structures of encoder and decoder need to be implemented for each experiment, see example experiments.
  - RKNTransitionCell the RKNTransitionCell as descirbed in the paper, implemented as sublcass of keras.layers.Layer in such a way that it can be used with keras.layers.RNN. 

### util
Utility functions

## Experiments

Currently Implemented:
  - Pendulum State Estimation (Implemented and verified that the ICML results are reproduced)
  - Pendulum Image Imputation (Implemented and verified that the ICML results are reproduced)
  - Quad Link State Estimation (Implemented)
  - Quad Link Image Imputation (Implemented)


## Dependencies

Tested with:
  - python 3.6
  - tensorflow 1.13.1 (both with and without GPU)
  - numpy 1.16
  - pillow 5.1.0 (only needed for data generation)

