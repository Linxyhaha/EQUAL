# Favor Warm Items? Equivariant Learning for Generalizable Cold-start Recommendation
This is the pytorch implementation of our paper
> Favor Warm Items? Equivariant Learning for Generalizable Cold-start Recommendation

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
python main.py --model_name=$1 --data_path=$2 --batch_size=$3 --l_r=$4 --reg_weight=$5 --num_neg=$6 --has_v=$7 --lr_lambda=$8 --num_sample=$9 --temp_value=$10 --dim_E=$11 --alpha=$12 --pos_ratio=$13 --neg_ratio=$14 --augment=$15 --aug_mode=$16 --align_all=$17 --mse_weight=$18 --sample=$19 --log_name=$20 --gpu=$21
```
or use run.sh
```
sh run.sh model_name data_path batch_size lr reg_weight num_neg True lr_lambda num_sample temp_value dim_E alpha pos_ratio neg_ratio augment aug_mode align_all mse_weight sample log_name gpu_id
```
- The log file will be in the './code/log/' folder. 
- The explanation of hyper-parameters can be found in './code/main.py'. 
- The default hyper-parameter settings are detailed in './code/hyper-parameters.txt'.