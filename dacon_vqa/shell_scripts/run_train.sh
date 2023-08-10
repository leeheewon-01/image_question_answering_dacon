NUM_GPU=3
GPU_IDS="0,1,2"
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
model_name_or_path="microsoft/git-large-coco"
CUDA_VISIBLE_DEVICES=$GPU_IDS \

for i in 0 1 2 3 4
do
    torchrun --nproc_per_node $NUM_GPU train.py \
        --output_dir "/home/user/4TB/hwlee/imgQA_output_$i" \
        --seed 42 \
        --model_name_or_path ${model_name_or_path} \
        --train_data_path "/home/hwlee/dacon/imgQA/split_df/split_train_$i.csv" \
        --valid_data_path "/home/hwlee/dacon/imgQA/split_df/split_valid_$i.csv" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 32 \
        --evaluation_strategy "steps" \
        --save_strategy "steps" \
        --logging_strategy "steps" \
        --logging_steps 5000 \
        --eval_steps 150 \
        --save_steps 150 \
        --save_total_limit 7 \
        --load_best_model_at_end \
        --learning_rate 5e-5 \
        --dataloader_num_workers 16 \
        --label_names "labels" \
        --fp16 \
        --remove_unused_columns "False" \
        --report_to='none' \
        --ddp_find_unused_parameters "False"
done