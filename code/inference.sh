nohup python -u inference.py --backmodel=$1 --drop_obj=$2 --dropout=$3 --topN=$4 --log_name=$5 --gpu=$6 >./log/DoubleCheck_$1_$2_$3_$4dropout_top$4_$5.txt 2>&1 &

# sh inference.sh CLCRec model [0,0.05,0.1,0.15,0.2] 100 log 0