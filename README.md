# Uncertainty aware Transformer

The code base for this is data2vec vision transformers.


## Classic VIT-B Pretraining and Finetuning

Command to pretrain the ViT-B model for 800 epochs
```

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=16 run_cyclical.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --log_dir ${OUTPUT_DIR} --num_mask_patches 120 \
        --model beit_base_patch16_224 \
        --seed 0 \
        --target_layers [6,7,8,9,10,11] \
        --ema_decay 0.9998 --ema_start_at 0 --ema_decay_init 0.999 \
        --batch_size 128 --lr 2e-3 --warmup_epochs 10 --epochs 800 \
        --clip_grad 3.0 --drop_path 0.25 --layer_scale_init_value 1e-4 \
        --layer_results 'end' \
        --var_w0 0.0 --var_w1 0.0 \
        --max_mask_patches_per_block 196 --min_mask_patches_per_block 16 \
        --l1_beta=2.0 \
        --weight_decay 0.05 \
        --imagenet_default_mean_and_std --dist_url $dist_url --loss_scale -1 --mask_dropout_prob -1.0 \
        --post_target_layer_norm --world_size 16 --attn_drop_rate 0.05 


```

Command to finetune the ViT-B model
```

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
        --model beit_base_patch16_224 \
        --finetune $CHECKPOINT \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --log_dir ${OUTPUT_DIR} --batch_size 128 --lr 4e-3 --update_freq 1 \
        --warmup_epochs 10 --epochs 100 --layer_decay 0.65 --drop_path 0.2 --drop 0.0 \
        --weight_decay 0.0 --mixup 0.8 --cutmix 1.0 --enable_deepspeed --nb_classes 1000 \
        --target_layer -1 --world_size 8 --dist_url $dist_url 

```
## Stochastic VIT-B Pretraining and Finetuning

Simply add the --stochastic flag 

```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=16 run_cyclical.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --log_dir ${OUTPUT_DIR} --num_mask_patches 120 \
        --model beit_base_patch16_224 \
        --seed 0 \
        --target_layers [6,7,8,9,10,11] \
        --ema_decay 0.9998 --ema_start_at 0 --ema_decay_init 0.999 \
        --batch_size 128 --lr 2e-3 --warmup_epochs 10 --epochs 800 \
        --clip_grad 3.0 --drop_path 0.25 --layer_scale_init_value 1e-4 \
        --layer_results 'end' \
        --var_w0 0.0 --var_w1 0.0 \
        --max_mask_patches_per_block 196 --min_mask_patches_per_block 16 \
        --l1_beta=2.0 \
        --weight_decay 0.05 \
        --imagenet_default_mean_and_std --dist_url $dist_url --loss_scale -1 --mask_dropout_prob -1.0 \
        --post_target_layer_norm --world_size 16 --attn_drop_rate 0.05 --stochastic
```

```

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
        --model beit_base_patch16_224 \
        --finetune $CHECKPOINT \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --log_dir ${OUTPUT_DIR} --batch_size 128 --lr 4e-3 --update_freq 1 \
        --warmup_epochs 10 --epochs 100 --layer_decay 0.65 --drop_path 0.2 --drop 0.0 \
        --weight_decay 0.0 --mixup 0.8 --cutmix 1.0 --enable_deepspeed --nb_classes 1000 \
        --target_layer -1 --world_size 8 --dist_url $dist_url --stochastic

```
