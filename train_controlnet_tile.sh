# conda activate diffusers
# screen -S tile_control_exp1; 202.168.109.223
MAX_STEPS=1000000
LR=1e-5
BS=4
PROMPT_DROPOUT=0.05
OUTPUT_DIR="controlnet/sd15_${BS}_${LR}_${MAX_STEPS}_dropout${PROMPT_DROPOUT}"
MODEL_NAME="/cephFS/yangying/AIGC2024/stable-diffusion-webui/models/Stable-diffusion/stable-diffusion-v1-5"

mkdir -p $OUTPUT_DIR


#screen -S tile_control_exp2; 202.168.101.119
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --multi_gpu train_controlnet_tile.py \
    --pretrained_model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --condition_resolution 64 \
    --resolution 512 \
    --learning_rate ${LR} \
    --max_train_steps ${MAX_STEPS} \
    --dataloader_num_workers 0 \
    --train_data_dir "/cephFS/video_lab/datasets/super_resolution/ffhq-dataset/images512x512_BLIPcaption/" \
    --validation_image "/cephFS/yangying/VSR2024/GPEN_223_RGB/val/lq_x4/ffhq_00000.png" "/cephFS/yangying/VSR2024/GPEN_223_RGB/val/lq_x4/ffhq_00001.png" \
    --validation_prompt "a infant baby boy" "a blonde lady with a big smile on her face " \
    --validation_steps 500 \
    --train_batch_size ${BS} \
    --resume_from_checkpoint latest \
    --mixed_precision "fp16" \
    --tracker_project_name "controlnet" \
    --report_to wandb \
    --proportion_empty_prompts ${PROMPT_DROPOUT} \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps 4 \
    --use_8bit_adam \
    --checkpointing_steps 1000 \
    # --set_grads_to_none
    # --checkpoints_total_limit=1000 \
    # --max_train_samples=80000 \