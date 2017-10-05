# What is Pints?

We are working on problems in the area of inverse modelling and baysian 
inference as applied to problems in electrochemistry and cardiac 
electrophysiology. Pints (Probabilistic Inference on Noisy Time-Series) 
is a collaborative repository to organise our efforts and share tools.

## How do you use pints

To stick a model in Pints, you need to make sure it implements two
methods:

```
dimension() --> Returns the dimension of the parameter space.
        
simulate(parameters, times) --> Returns a vector of model evaluations
  at the given times, using the given parameters
```

If your model implements these methods - or you can write a wrapper
class that does - you can start using Pints for optimisation or mcmc
inference methods!

Examples are given in the `example_x.py` files, in the root of the
project.

## Testing pints

To run all tests, lone the repository, navigate to the pints directory
and type:

```
python -m unittest discover -v test
```

Or use the bash script `run-tests.sh`


