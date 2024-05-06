accelerate launch --main_process_port 29500 train.py \
    --learning_rate=1e-4 \
    --pretrained_adapter_path="YZBPXX/CCDA" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
    --run_name="test" \
    --num_tokens=16 \
    --num_train_epochs=100000000 \
    --train_batch_size=1 \
    --eval_step=2000
