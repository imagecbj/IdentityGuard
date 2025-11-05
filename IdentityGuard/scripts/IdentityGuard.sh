#!/bin/bash

export GPU=0
# export DATASET="CelebA-HQ"
export DATASET="VGGFace2"
export MODEL_PATH="./stable-diffusion-2-1-base"
export CLASS_DIR="data/class-person"

params=(
    "0.1 0.5 1"
)

process=(
""
)

for p in "${process[@]}"; do

for param_tuple in "${params[@]}"; do
    read -r LATENT ATT NOISE <<< "$param_tuple"

    export LATENT ATT NOISE
    export EXPERIMENT_NAME="latent${LATENT}_reverse${ATT}_noise${NOISE}_sel_mss_ada_all_attack"

    # Set paths (including experiment name)
    export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs${GPU}/${EXPERIMENT_NAME}"
    export DREAMBOOTH_after0_DIR="dreambooth-after-outputs${GPU}/${EXPERIMENT_NAME}"
    export DREAMBOOTH_after1_DIR="dreambooth-after-outputs${GPU}/${EXPERIMENT_NAME}/checkpoint-1000"
    export DREAMBOOTH_DIR="$DREAMBOOTH_OUTPUT_DIR/checkpoint-1000"

    echo "🔧 Starting parameter set: latent=$LATENT att=$ATT noise=$NOISE"
    echo "💡 Experiment name: $EXPERIMENT_NAME"
    echo "🖼️ Current process type: $p"

    i=0
    start_idx=1
    end_idx=50
    for PERSON_DIR in data/$DATASET/*; do
        i=$((i+1))

        if [ "$i" -lt "$start_idx" ] || [ "$i" -gt "$end_idx" ]; then
            continue
        fi

        PERSON=$(basename "$PERSON_DIR")

        # Set path variables
        export CLEAN_ADV_DIR="data/$DATASET/$PERSON/set_A"
        export CLEAN_TRAIN_DIR="data/$DATASET/$PERSON/set_B"
        export OUTPUT_DIR="outputs/${EXPERIMENT_NAME}/$DATASET/$PERSON/set_A"
        export INSTANCE_DIR="$OUTPUT_DIR/"
        export INFER_OUTPUT_DIR="txt2img-samples${p}/${EXPERIMENT_NAME}/$DATASET/$PERSON/set_A"

        echo "🧪 Processing sample: $PERSON (index $i)"

        # === Step 1: Generate adversarial samples ===
        if [ ! -d "$OUTPUT_DIR" ]; then
            echo "🏋️ Starting model fine-tuning..."
            accelerate launch train_dreambooth.py \
                --pretrained_model_name_or_path=$MODEL_PATH \
                --enable_xformers_memory_efficient_attention \
                --train_text_encoder \
                --instance_data_dir=$CLEAN_ADV_DIR \
                --class_data_dir=$CLASS_DIR \
                --output_dir=$DREAMBOOTH_after0_DIR \
                --with_prior_preservation \
                --prior_loss_weight=1.0 \
                --instance_prompt="a photo of sks person" \
                --class_prompt="a photo of person" \
                --resolution=512 \
                --train_batch_size=1 \
                --gradient_accumulation_steps=2 \
                --learning_rate=5e-7 \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --num_class_images=200 \
                --max_train_steps=1000 \
                --checkpointing_steps=1000 \
                --center_crop \
                --mixed_precision=bf16 \
                --prior_generation_precision=bf16 \
                --sample_batch_size=8

            echo "⚔️ Running adversarial attack..."
            accelerate launch attacks/IdentityGuard.py \
                --pretrained_model_name_or_path=$MODEL_PATH \
                --after_pretrained_model_name_or_path=$DREAMBOOTH_after1_DIR \
                --enable_xformers_memory_efficient_attention \
                --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
                --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
                --instance_prompt="a photo of sks person" \
                --class_data_dir=$CLASS_DIR \
                --num_class_images=200 \
                --class_prompt="a photo of person" \
                --output_dir=$OUTPUT_DIR \
                --center_crop \
                --prior_loss_weight=1.0 \
                --resolution=512 \
                --train_text_encoder \
                --train_batch_size=1 \
                --max_train_steps=50 \
                --max_f_train_steps=3 \
                --max_adv_train_steps=6 \
                --checkpointing_iterations=50 \
                --learning_rate=5e-7 \
                --pgd_alpha=5e-3 \
                --pgd_eps=5e-2 \
                --with_prior_preservation \
                --use_cosin_sim \
                --sim_param=$LATENT \
                --att_param=$ATT \
                --noise_param=$NOISE \
                --save_dir="att_map_save/$EXPERIMENT_NAME" \
                --loss_func="reverse_all" \
                --use_MSS \
                --use_CAE \
                --use_search \
                --seed=0
        else
            echo "✅ Adversarial samples already exist, skipping generation"
        fi

        # === Step 2+3: Fine-tune model and generate inference samples ===
        if [ ! -d "$INFER_OUTPUT_DIR" ]; then
            echo "🏋️ Starting model fine-tuning..."
            accelerate launch train_dreambooth.py \
                --pretrained_model_name_or_path=$MODEL_PATH \
                --enable_xformers_memory_efficient_attention \
                --train_text_encoder \
                --instance_data_dir=$INSTANCE_DIR \
                --class_data_dir=$CLASS_DIR \
                --output_dir=$DREAMBOOTH_OUTPUT_DIR \
                --with_prior_preservation \
                --prior_loss_weight=1.0 \
                --instance_prompt="a photo of sks person" \
                --class_prompt="a photo of person" \
                --resolution=512 \
                --train_batch_size=1 \
                --gradient_accumulation_steps=2 \
                --learning_rate=5e-7 \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --num_class_images=200 \
                --max_train_steps=1000 \
                --checkpointing_steps=1000 \
                --center_crop \
                --mixed_precision=bf16 \
                --prior_generation_precision=bf16 \
                --sample_batch_size=8

            echo "🎨 Generating inference samples..."
            CUDA_VISIBLE_DEVICES=${GPU} python infer.py \
                --model_path=$DREAMBOOTH_DIR \
                --output_dir=$INFER_OUTPUT_DIR
        else
            echo "✅ Inference samples already exist, skipping generation"
        fi
    done
done
done

echo "🎉 All parameter combination experiments completed!"
