# Profiling Categorical Binary Base-Learner

## Data Simulation

- 5x10e5 observations
- 20 classes of equal size

Therefore, we have a size of 76.29 for the raw data in binary form.

## Considerations Binary vs. Linear Base-Learner

- Binary stores data as sparse matrix, linear base-learner does not

## Valgrind

Profiling is done with [massif](https://www.valgrind.org/docs/manual/ms-manual.html) for stack and heap. To visualize results we use the [massif-visualizer](https://kde.org/applications/development/org.kde.massif-visualizer).

Note that running R in debugging mode with massif speeds down the execution 20 to 50 times.
