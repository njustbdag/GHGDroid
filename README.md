# GHGDroid
GHGDroid : Global Heterogeneous Graph-based Android Malware Detection

## Introduction of Individual Files
1. args.py : saves related parameters
2. build_graph.py : generate GHG in json format
3. extract.py : extract api invocation from smali folders
4. get_api_data : count and match sapi, generate api corpus for skip-gram model and calculate sensitive coefficient
5. model.py : the structure of iterative aggregated gcn stacked network model
6. process.py : preprocess the data before it was fed into model
7. sampling.py : neighbor sampling on the GHG
8. utils : tools help the experiment

## Execution of Experiments
1. generate related data before training with File 2,3,4 
2. set the parameter in File 1
3. run main.py

## Attention
We have stored the intermediate product data for a particular experiment in various folders, and this code can be run directly if you want to.
If you wish to run the code on your own dataset, you will need to regenerate these intermediate product data
Just follow the above steps, you can run the code on your own dataset
