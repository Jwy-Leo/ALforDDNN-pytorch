Epochs=320
dataset="SmartHome"
ConfigPath="config/Abs_config.yaml"
Begin_num=2
Total_num=2

for i in $(seq 0 3)
do
    GPU_index=$(expr $i % $Total_num)
    BGPU_index=$(expr $Begin_num + $GPU_index)
    export CUDA_VISIBLE_DEVICES=$BGPU_index
    if [ $GPU_index != $(expr $Total_num - 1) ]
    then
        #echo $i
        python main_AL.py --dataset $dataset --Epochs $Epochs --config_path $ConfigPath &
        sleep 5s
    else
        python main_AL.py --dataset $dataset --Epochs $Epochs --config_path $ConfigPath
    fi
done
