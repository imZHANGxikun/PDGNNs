[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This is the official repository of the KDD 2024 paper 'Topology-aware Embedding Memory for Continual Learning on Expanding Networks'

 ## Get Started
 
 This repository contains the PDGNNs-TEM framework implemented for GPU devices. The following packages are required to run the code:
 
* python==3.7.10
* scipy==1.5.2
* numpy==1.19.1
* torch==1.7.1
* networkx==2.5
* scikit-learn~=0.23.2
* matplotlib==3.4.1
* ogb==1.3.1
* dgl==0.6.1
* dgllife==0.2.6
 

 ## Code Usages
 
To run all the experiments under the class-IL setting with three different strategies for TE generation, please run:

 ```
 bash run_all_exps_classIL.sh 
 ```

To run all the experiments under the task-IL setting with three different strategies for TE generation, please run:

 ```
 bash run_all_exps_taskIL.sh 
 ```
