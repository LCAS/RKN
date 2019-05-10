# Recurrent Kalman Networks
run... files run a single experiment

exlaunch... files launch multiple experiments (only make sense on the cluster and only if you use the right version of ezex)

## Run existing experiments

Command line flags can be used to configure the run. Not all parameters are configurable via flags.
New flags can be added and the value passed to the config class via the constructor (see existing flags for example).

Detailed description of all setable parameters can be found in model/RKNConfig.py

## Currently Working Experiments:

### Spring Damper:
Linear Spring damper system:
   - Spring Damper: 1 or 2 Dimensional spring damper, with (noisy) position as observation
   - Spring Damper Visual: 1-Dimensional spring damper with very easy observation. The overfitting is due to the encoder.

### Pendulum:
There is transition noise on the angular velocity of the pendulum.

   - Pendulum Constant Noise: Track ball mounted to endeffector of (invisible) pendulum. Constatn observation noise on ball position
   - Pendulum Image Noise: Track angle (or rather sin and cos of angle) of pendulum. There is noise on the image (interpolation bewtween pure noise and true image, factors changing but correlated over time.)

## Currently Not Working Experiments:
Worked before, but config and run... not up to date:
   - Ball Tracking (linear, dual and quad link)
   - Toy Block Tower

### Other


- Kitti - only preliminary implementation - not sufficiently tested or evaluated.
- Bitcoin - the bitcoin experiments were implemented for another project in the course of the PGM lecture @ TU Darmstadt. Obviously never worked, or I would not be here anymore ;-)



## Libraries and Environment
Current Version Tested with
- Python 3.5.2
- Tensorflow 1.7 (with and without CUDA)
- Numpy 1.13.1

Some experiments need additional python packages
- Matplotlib 1.5.3 (all Experiments)
- OpenCV (cv2) (only for the old ball data - Its planned to get rid of this)
- PIL 4.1.0 (only Pendulum and Ball Tracking)
- SciPy (only ToyBlockTower and KITTI)
- pykitti (https://github.com/utiasSTARS/pykitti) (only KITTI)
- pykalman (only for spring damper and pendulum)

## Module Description

### rkn
The RKN Model

### baselines
Models for Baseline Transition Cells (LSTM and GRU)

#### config
The Config files for the implemented experiments

#### data
Loading and preparing the data.

**Ball Data**:
Apparently I broke something while cleaning up the code - for now the old Ball data is used:
- Balls instead of LinearBalls
- DLBalls instead of DoubleLinkBalls
- QLBalls instead of QuadLinkBalls

#### model
The main RKN code:
- RKN: Builds the model (i.e. the tensorflow graph)
- RKNConfig: Holds the parameters to configure the RKN, this class is abstract and needs to be implemented for each experiment.
- RKNLinear: Special version of RKN, without encoder and decoder, only usable for linear models in true state space (e.g. spring damper)
- RKNRunner: Runs the model (i.e. executes the tensorflow graph) provides methods for training, evaluation and prediction

#### model_dataset
**Probably won't work, not keept up to date**

The main RKN code, adapted to work with the new tensorflow input pipeline (tf.data.Dataset).
This is not properly tested and only works for Tensorflow 1.4 - use with caution (if at all)

#### model_local_linear
Code for the Local Linear RKN including:
- LLRKN: Analogously to RKN but builds locally linear transition model
- LLRKNConfig: Analogously to RKNConfig but with few additional values

Note that the "normal" RKNRunner can (and should) be used to execute the model

#### network
Helper and "Toolbox" for feed forward networks (used to construct encoder and decoder)

#### plotting
Code for plotting and visualization

#### preprocessor
Preprocessors that can be given to the model - applied to inputs befor fed trough network

#### transition_cell
Special RKN Transition Cell:
- TransitionCell: abstract base
- RKNSimpleTransitionCell: RKN Transition cell with **uncorrelated** covariances
- RKNCorrTransitionCell: RKN Transition cell with **correlated** covariances
- LLRKNCorrTransitionCell: Localy Linear RKN Transition cell with **correlated** covariance
- RKNFullTransitionCell: RKN Transition cell with **full** covariances (i.e. a Kalman Filter)
#### util
Utility functions and classes

### n\_link\_sim
Simulator for the Dual and Quad Link (written in C, see seperate readme for more information)

## Implement your own experiment

1.) Write a config class (lets call it 'MyConfig') for your experiment.
This class needs to be a subclass of model.RKNConfig.
See model.RKNConfig for documentation and the config for the existing experiments for examples.

2.) Create config object from previously created class
```python
config = MyConfig(args)
```
3.) Create model.RKN (the model) and model.RKNRunner (executes the model)
```python
model = RKN(name, config)
model_runner = RKNRunner(model)
```
4.) Train and evaluate model using
```python
 model_runner.train(args)
 model_runner.evaluate(args)
```
5.) Use model for inference
```python
 model_runner.predict(args)
```

For further documentation and 'args' see docstrings in code