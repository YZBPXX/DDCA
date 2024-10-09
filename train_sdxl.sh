accelerate launch --main_process_port 29500 train.py \
    --learning_rate=1e-4 \
    --pretrained_ip_adapter_path="/hy-tmp/models/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin" \
    --pretrained_model_name_or_path="/hy-tmp/models/stable-diffusion-xl-base-1.0" \
    --image_encoder_path="/hy-tmp/models/IP-Adapter/models/image_encoder" \
    --run_name="exp1" \
    --num_tokens=16 \
    --num_train_epochs=100000000 \
    --train_batch_size=1 \
    --eval_step=2000
