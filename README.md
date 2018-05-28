# wing-loss

181787 train + val
19880 test


python create_tfrecords.py \
    --image_dir=/home/gpu2/hdd/dan/CelebA/val/images/ \
    --annotations_dir=/home/gpu2/hdd/dan/CelebA/val/annotations/ \
    --output=data/val_shards/ \
    --num_shards=100
    
python create_tfrecords.py \
    --image_dir=/home/gpu2/hdd/dan/CelebA/train/images/ \
    --annotations_dir=/home/gpu2/hdd/dan/CelebA/train/annotations/ \
    --output=data/train_shards/ \
    --num_shards=500