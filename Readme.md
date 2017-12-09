```
                                                     ___.                          __            
                            ____  ____   _____ ______\_ |__   ____   ____  _______/  |_          
                          _/ ___\/  _ \ /     \\____ \| __ \ /  _ \ /  _ \/  ___/\   __\         
                          \  \__(  <_> )  Y Y  \  |_> > \_\ (  <_> |  <_> )___ \  |  |           
                           \___  >____/|__|_|  /   __/|___  /\____/ \____/____  > |__|           
                               \/            \/|__|       \/                  \/                 
```

# Installation

# Idea

# Usage

To play around with the classes there are some "tutorial" `R` files within the tutorial folder. The naming corresopnds
to the class or function which we want to make clearer (e.g. `tutorial/loss_class.R`).

# Implementation

## Compatibility of the C++ Code for other Languages

The core features are purely written in `C++`. This should give the opportunity to use the implementation in a wider range
than just for `R`. The restrictions here which have to be covered bevore exporting the code:

- Use of `R`s `RcppArmadillo` instead of pure `Armadillo`. This is just for convenience and should be replaceable by the original
  `Armadillo` library. Hence, the linking to `BLAS` and `Lapack` has to be done manually.
  
- In some files we use the `Rcpp` class `Function` to give the ability of using custom `R` losses or baselearner within the code.
  This has to be replaced or completely dropped out before exporting the code.
  
- The wrapper files are not parts of the main `C++` implementation. They are just used to export the classes to `R`. Therefore the
  user has the total control about the classes. In addition, unit testing is more convenient.

# References
