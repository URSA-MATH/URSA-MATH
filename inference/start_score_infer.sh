#!/bin/bash
dataset_name='mathverse'
model_path=''
data_path=''
image_root=''
work_dir=''
cuda_sum=1

logs_dir="${work_dir}/logs"
if [ ! -d "$logs_dir" ]; then
    echo "Creating logs directory: $logs_dir"
    mkdir -p "$logs_dir"
fi

selects_dir="${work_dir}/selects"
if [ ! -d "$selects_dir" ]; then
    echo "Creating selects directory: $selects_dir"
    mkdir -p "$selects_dir"
fi

for i in $(seq 0 $(($cuda_sum - 1))); do

    output_path="${selects_dir}/select_answer_device_${i}.pt"
    cuda_device=$i
    log_file="${logs_dir}/select_answer_${dataset_name}_${i}.log"

    echo "Running inference on GPU $cuda_device."
    echo "Save output at $output_path."
    nohup python3 inference/prm_infer_score.py \
        --output_path $output_path \
        --dataset_name $dataset_name \
        --model_path $model_path \
        --data_path $data_path \
        --image_root $image_root \
        --work_dir $work_dir \
        --cuda_sum $cuda_sum \
        --best_of_n 64 \
        --cuda_device $cuda_device \
        --dtype torch.bfloat16 > $log_file 2>&1 &

    sleep 4
done

wait

echo "All tasks completed."
