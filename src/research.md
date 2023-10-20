# Part 2. Research

|                   |    10 runs |   100 runs |  1000 runs  | Average time of 1 run|
|-------------------|------------|------------|-------------|----------------------|
| Matrix perceptron | 1.955 sec | 19.642 sec  | 195.378 sec |  0.195  sec          |
| Graph perceptron  | 1.871 sec | 18.663 sec  | 187.947 sec |  0.188  sec          |


The experiment on the test sample (14800 test) is done 10 times, 100 times and 1000 times.

The best weights are used (file `weights_2_784_86__.txt` with 86.59% average accuracy). 

The difference does not look significant (less than 5 percent), but the graph realization is slightly faster.

Test stand specification:  
-   CPU: Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz x 6
-   RAM: 24 GB
