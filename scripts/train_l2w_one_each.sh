#!/bin/bash
MODEL="Local2WorldModel(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
mv_dec1='MultiviewDecoderBlock_max',mv_dec2='MultiviewDecoderBlock_max', enc_minibatch = 12, need_encoder=True)"

# TRAIN_DATASET="4000 @ ScanNetpp_Seq(filter=True, sample_freq=3, num_views=13, split='train', aug_crop=256, resolution=224, transform=ColorJitter, seed=233) + \
# 2000 @ Aria_Seq(num_views=13, sample_freq=2, split='train', aug_crop=128, resolution=224, transform=ColorJitter, seed=233) + \
# 2000 @ Co3d_Seq(num_views=13, sel_num=3, degree=180, mask_bg='rand', split='train', aug_crop=16, resolution=224, transform=ColorJitter, seed=233)"
# TEST_DATASET="1000 @ ScanNetpp_Seq(filter=True, sample_freq=3, num_views=13, split='test', resolution=224, seed=666)+ \
# 1000 @ Aria_Seq(num_views=13, split='test', resolution=224, seed=666) + \
# 1000 @ Co3d_Seq(num_views=13, sel_num=3, degree=180, mask_bg='rand', split='test', resolution=224, seed=666)"

TRAIN_DATASET="[ 1000 @ SevenScenes_Seq(
            scene_id='chess',
            set_class='train',
            seq_id=1,
            num_views=3,
            sample_freq=20,
            start_freq=1,
            resolution=(224, 224)), 
1000 @ SevenScenes_Seq(
            scene_id='fire',
            set_class='train',
            seq_id=1,
            num_views=3,
            sample_freq=20,
            start_freq=1,
            resolution=(224, 224)), 
1000 @ SevenScenes_Seq(
            scene_id='heads',
            set_class='train',
            seq_id=2,
            num_views=3,
            sample_freq=20,
            start_freq=1,
            resolution=(224, 224)),
1000 @ SevenScenes_Seq(
            scene_id='office',
            set_class='train',
            seq_id=1,
            num_views=3,
            sample_freq=20,
            start_freq=1,
            resolution=(224, 224)),
1000 @ SevenScenes_Seq(
            scene_id='office',
            set_class='train',
            seq_id=3,
            num_views=3,
            sample_freq=20,
            start_freq=1,
            resolution=(224, 224)),
1000 @ SevenScenes_Seq(
            scene_id='pumpkin',
            set_class='train',
            seq_id=2,
            num_views=3,
            sample_freq=20,
            start_freq=1,
            resolution=(224, 224)),
1000 @ SevenScenes_Seq(
            scene_id='redkitchen',
            set_class='train',
            seq_id=1,
            num_views=3,
            sample_freq=20,
            start_freq=1,
            resolution=(224, 224)),
500 @ SevenScenes_Seq(
            scene_id='stairs',
            set_class='train',
            seq_id=2,
            num_views=3,
            sample_freq=20,
            start_freq=1,
            resolution=(224, 224)) ]"


TEST_DATASET="[1000 @ SevenScenes_Seq(
            scene_id='chess',
            set_class='test',
            seq_id=3,
            num_views=3,
            sample_freq=20,
            start_freq=1,
            resolution=(224, 224))]"

PRETRAINED="checkpoints/slam3r_l2w.pth"
TRAIN_OUT_DIR="checkpoints/slam3r_7scenes_l2w_ten_epo"

python train.py \
    --train_dataset "${TRAIN_DATASET}" \
    --test_dataset "${TEST_DATASET}" \
    --model "$MODEL" \
    --train_criterion "Jointnorm_ConfLoss(Jointnorm_Regr3D(L21,norm_mode=None), alpha=0.2)" \
    --test_criterion "Jointnorm_Regr3D(L21, norm_mode=None)" \
    --pretrained $PRETRAINED \
    --pretrained_type "slam3r" \
    --lr 5e-5 --min_lr 5e-7  --epochs 10 --batch_size 4 --accum_iter 2 \
    --save_freq 2 --keep_freq 20 --eval_freq 1 --print_freq 100\
    --save_config\
    --output_dir $TRAIN_OUT_DIR \
    --freeze "encoder"\
    --loss_func "l2w" \
    --ref_ids 0

