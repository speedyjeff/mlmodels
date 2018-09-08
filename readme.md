# Example use of ML.NET and OpenCV

I needed a prediction model for my small game so I gave both ML.NET and OpenCV a try.  
ML.NET's FastRegressionTree outperformed OpenCV's RandomTree, but the performance difference was magnititudes better for OpenCV. 

## Feedback

_ |ML.NET | OpenCV
-----|-------|-------
X-Plat| Yes | (OpenCVSharp-AnyCPU)[https://www.nuget.org/packages/OpenCvSharp3-AnyCPU/] only has native binaries for Windows X86/X64, though OpenCV is x-plat
Prediction Throughput | Average prediction is 3ms | Average prediction is 10us (0.01ms)
On disk size | Small compressed | large raw text (compressable)
Feedback | Create/destroy threads per prediction (extreme slow down when debugging, due to serializing output to the debug output window) | Cumbersome GCHandle to correctly pin input/output
_ | STDOUT output when training | Use of [,] arrays (need to convert data often)
_ | Not able to use intillisense to build pipeline (everything takes an ILearningPipelineItem) | Limited set of models
_ | Limited set of models | Training is slow
_ | Loading can be slow |

## Results

_       | ML.NET (ms)   | OpenCV (ms)   | ML.NET/OpenCV
--------|---------------|---------------|-----------
Training   |    4137.00    |   36131.00    |       0.11
Saving    |      62.00    |      34.00    |       1.82
Size on disk    |    3873.00    |  466599.00    |       0.01
Loading    |      94.00    |      33.00    |       2.85
Evaluating    |     624.00    |     449.00    |       1.39
R^2     |        NaN    |       0.45    |        NaN
Preding    |    2492.00    |       8.00    |     311.50
Preding/Iteration |       2.49    |       0.01    |     311.50
Parallel Prediction  |   11771.00    |      22.00    |     535.05

