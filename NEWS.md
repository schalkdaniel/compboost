## compboost 0.1.0

Initial release

- **19.07.2018** \
  Compboost now uses sparse matrices for splines to reduce memory load.

- **29.06.2018** \
  Compboost API is almost ready to use.
  
- **14.06.2018** \
  Update naming `GreedyOptimizer` -> `OptimizerCoordinateDescent` and small typos.

- **30.03.2018** \
  Compboost is now ready to do binary classification by using the 
  `BernoulliLoss`.
  
- **29.03.2018** \
  Upload `C++` documentation created by doxygen. 

- **28.03.2018** \
  P-Splines are now availalbe as baselearner. Additionally the Polynomial and P-Spline learner
  are speeded up using a more gneeral data structure which stores the inverse once and reuse it for
  every iteration.

- **21.03.2018** \
  New data structure with independent source and target.
  
- **01.03.2018** \
  Compboost should now run stable and without memory leaks.

- **07.02.2018** \
  Naming of the `C++` classes. Those are matching the `R` classes now.

- **29.01.2018** \
  Update naming to a mroe consistent scheme.
  
- **26.01.2018** \
  Add printer for the classes.
  
- **22.01.2018** \
  Add inbag and out of bag logger.
  
- **21.01.2018** \
  New structure for factorys and baselearner. The function
  `InstantiateData` is now member of the factory, not the baselearner. This 
  should also speed up the algorithm, since we don't have to check whether data
  is instantiated or not. We can do that once within the constructor. 
  Additionally, it should be more clear now what the member does since there is
  no hacky baselearner helper necessary to instantiate the data.
