# Developing... 
# Gradient Server

- Background: In the case of low bandwidth, the communication time consumption of A-SGD will result in slower training of the model, and the increase of staleness will result in the convergence of the model is not guaranteed.

- Basic idea: Temporary confidential.

- Experimental results: Implemented ParameterServer using distributed API of PyTorch. Train AlexNet at 100Mbps bandwidth. Our new training mechanism is 10 times faster than the traditional A-SGD and has better convergence effect under the same number of iterations.

- Further optimization: Accelerate the server with multi-process computing and multi-thread communication, which solves the performance bottleneck caused by the gradient sparse operation. Add a "pseudo-synchronization" mechanism to ensure convergence.
