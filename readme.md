# Example use of ML.NET and OpenCV

I needed a prediction model for my small game so I gave both ML.NET and OpenCV a try.  
ML.NET's FastRegressionTree universally had better fitness than OpenCV's RandomTree.  But ML.NET's perdiction throughput was two magnititudes slower than OpenCV. (SEE UPDATE BELOW)

## UPDATE - using Microsoft.ML version 0.7
The performance of Microsoft.ML is competitive and sometimes better.

### Feedback

_ |ML.NET | OpenCV
-----|-------|-------
X-Plat| Yes | [OpenCVSharp-AnyCPU](https://www.nuget.org/packages/OpenCvSharp3-AnyCPU/) only has native binaries for Windows X86/X64, though OpenCV is x-plat
Prediction Throughput | **Average prediction is 20us (0.02ms)** | Average prediction is 10us (0.01ms)
On disk size | **Small compressed** | large raw text (compressable)
Thread safe | No | No
Model load time | **Generally slow** | Relatively fast (regardless of model size)
Algorithm select | (did not reevaluate) | Slightly larger but limited
Documentation/Samples | Handful | Large community of examples
Feedback | Need for duplicated code when load, save, predict are done in seperate functions | Cumbersome GCHandle to correctly pin input/output
_ | STDOUT output when training | Use of [,] arrays (need to convert data often)
_ | Not able to use intillisense to build pipeline (everything takes an ILearningPipelineItem) | Training is slow
_ | Typically the FastTreeRegressor has better fitness than RandomTrees | 


### Results
_       | ML.NET (ms)   | OpenCV (ms)   | ML.NET/OpenCV
--------|---------------|---------------|-----------
Training   |    **2857.00**    |   22219.00    |       0.13
Saving    |     187.00    |      **24.00**    |       7.79
Size on disk    |   **81805.00**    |  466599.00    |       0.18
Loading    |     213.00    |      **18.00**    |      11.83
Evaluating    |     943.00    |     **193.00**    |       4.89
R^2     |       **0.66**    |       0.45    |       1.47
Predicting (1000 itrs)    |      15.00    |       **3.00**    |       5.00
Prediction/Iteration |       0.02    |       **0.00**    |       5.00
Parallel Prediction  |      **14.00**    |      16.00    |       0.88

## using Microsoft.ML version 0.5

### Feedback

_ |ML.NET | OpenCV
-----|-------|-------
X-Plat| Yes | [OpenCVSharp-AnyCPU](https://www.nuget.org/packages/OpenCvSharp3-AnyCPU/) only has native binaries for Windows X86/X64, though OpenCV is x-plat
Prediction Throughput | **Average prediction is 3ms** | Average prediction is 10us (0.01ms)
On disk size | Small compressed | large raw text (compressable)
Thread safe | No | No
Model load time | **Generally slow** | Relatively fast (regardless of model size)
Algorithm select | Limited | Slightly larger but limited
Documentation/Samples | Handful | Large community of examples
Feedback | **Create/destroy threads per prediction (extreme slow down when debugging, due to serializing output to the debug output window)** | Cumbersome GCHandle to correctly pin input/output
_ | STDOUT output when training | Use of [,] arrays (need to convert data often)
_ | Not able to use intillisense to build pipeline (everything takes an ILearningPipelineItem) | Training is slow
_ | Typically the FastTreeRegressor has better fitness than RandomTrees | 

### Results

#### 200,000 rows, 36mb
_       | ML.NET (ms)   | OpenCV (ms)   | ML.NET/OpenCV
--------|---------------|---------------|-----------
Training   |    3611.00    |   28995.00    |       0.12
Saving    |      58.00    |      17.00    |       3.41
Size on disk    |    81308.00    |  466599.00    |       0.17
Loading    |      65.00    |      **17.00**    |       3.82
Evaluating    |     472.00    |     243.00    |       1.94
R^2     |        **0.66**    |       0.45    |       1.47
Predicting (1000 itrs)    |    1874.00    |       **4.00**    |     468.50
Predicting/Iteration |       1.87    |       0.00    |     468.50
Parallel Prediction  |   9205.00    |      12.00    |     767.08

#### 3,240,000 rows, 410mb
_       | ML.NET (ms)   | OpenCV (ms)   | ML.NET/OpenCV
--------|---------------|---------------|-----------
Training   |    22228.00    |  627119.00    |       0.04
Saving    |      191.00    |      44.00    |       4.34
Size on disk    |    79863.00    |  459635.00    |       0.17
Loading    |      99.00    |      **16.00**    |       6.19
Evaluating    |     435.00    |     183.00    |       2.38
R^2     |        **0.80**    |       0.45    |       1.78
Predicting (1000 itrs)    |    1477.00    |       **3.00**    |     492.33
Predicting/Iteration |       1.48    |       0.00    |     492.33
Parallel Prediction  |   7265.00    |      29.00    |     250.52

