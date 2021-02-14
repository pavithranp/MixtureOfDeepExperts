# MixtureOfDeepExperts

This is implementation is loosely based on the paper "Choosing smartly". It incorporates multiple CNN detectors "Experts"and combines their output using a gated weighting network. 

## Sample outputs:
| RGB channel    | Depth Channel       | Gating network |
| -------------- | -------------- | ---------|
![yos1](output/rgb_1.png) |   ![yos2](output/depth.png)|![yos2](output/gating.png)| 



## Dependencies :

* detectron2
* pytorch 1.7.0 + cuda 11.0
* [InOutDoorPeople dataset](http://adaptivefusion.cs.uni-freiburg.de/#dataset)

