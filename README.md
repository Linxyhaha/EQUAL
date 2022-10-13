# Favor Warm Items? Equivariant Learning for Out-of-Distribution Cold-start Recommendation
Cold-start Recommendation
This is the pytorch implementation of our paper
> Favor Warm Items? Equivariant Learning for Out-of-Distribution Cold-start Recommendation

## Environment
- Anaconda 3
- python 3.7.11
- pytorch 1.10.0
- numpy 1.21.4

## Usage
### Data
The experimental data are in './data' folder.

### Training
```
python main.py
```
or use run.sh
```
sh run.sh
```
- The log file will be in the './code/log/' folder. 
- The explanation of hyper-parameters can be found in './code/main.py'. 
- The default hyper-parameter settings are detailed in './code/hyper-parameters.txt'.