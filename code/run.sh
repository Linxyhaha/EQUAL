nohup python -u main.py --has_t=True --model_name=$1 --data_path=$2 --batch_size=$3 --l_r=$4 --reg_weight=$5 --num_neg=$6 --lr_lambda=$7 --num_sample=$8 --temp_value=$9 --dim_E=$10 --alpha=$11 --pos_ratio=$12 --neg_ratio=$13 --align_all=$14 --mse_weight=$15 --log_name=$16 --gpu=$17 >./log/$1_$2_$3bs_$4lr_$5reg_$6ng_$7lambda_$8rou_$9temp_$10dimE_$11alpha_$12pos_$13neg_$14all_$15mse_$16.txt 2>&1 &

# Example
# sh run.sh CLCRec micro-video 256 0.001 0.001 64 0.1 0.7 1 128 0.9 0.1 0.1 0 0.01 log 0

# please remove "--has_v=True" in command if you are using datasets with no textual features, e.g., amazon, kwai.
# sh run.sh CLCRec amazon 256 0.001 0.001 256 0.1 0.5 0.1 128 0.9 0.1 0.2 0 1 log 0
# sh run.sh CLCRec kwai 512 0.001 0.001 512 0.1 0.3 1 128 0.8 0.7 0.2 0 1 log 0