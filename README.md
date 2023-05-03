# Equivariant Learning for Out-of-Distribution Cold-start Recommendation
This is the pytorch implementation of our paper
> Equivariant Learning for Out-of-Distribution Cold-start Recommendation

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
python main.py --model_name=$1 --data_path=$2 --batch_size=$3 --l_r=$4 --reg_weight=$5 --num_neg=$6 --lr_lambda=$7 --num_sample=$8 --temp_value=$9 --dim_E=$10 --alpha=$11 --pos_ratio=$12 --neg_ratio=$13 --align_all=$14 --mse_weight=$15 --log_name=$16 --gpu=$17
```
or use run.sh
```
sh run.sh CLCRec micro-video 256 0.001 0.001 512 0.1 0.7 1 128 0.9 0.1 0.1 0 0.01 log 0
```
- The log file will be in the './code/log/' folder. 
- The explanation of hyper-parameters can be found in './code/main.py'. 
- The default hyper-parameter settings are detailed in './code/hyper-parameters.txt'.

### Inference
Get the results of EQUAL with Implicit Alignment Module (IAM) by running inference.py:
```
python inference.py --backmodel=$1 --drop_obj=$2 --dropout=$3 --topN=$4 --log_name=$5 --gpu=$6
```
or use inference.sh
```
sh inference.sh CLCRec model [0,0.05,0.1,0.15,0.2] 100 log 0
```
The explanation of hyper-parameters can be found in './code/inference.py'. 
The default hyper-parameter settings are detailed in './code/hyper-parameters.txt'.

### Examples

1. Train EQUAL on micro-video:

```
cd ./code
sh run.sh CLCRec micro-video 256 0.001 0.001 512 0.1 0.7 1 128 0.9 0.1 0.1 0 0.01 log 0
```

2. Inference on Amazon:

```
cd ./code
python inference.py --backmodel CLCRec --drop_obj model --dropout [0,0.05,0.1,0.15,0.2] --topN 50 --log_name log --gpu 0
```