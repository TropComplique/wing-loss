# wing-loss

174502 train + val
19079 test


python create_tfrecords.py \
    --image_dir=/home/dan/datasets/CelebA/val/images/ \
    --annotations_dir=/home/dan/datasets/CelebA/val/annotations/ \
    --output=data/val_shards/ \
    --num_shards=100
    
python create_tfrecords.py \
    --image_dir=/home/dan/datasets/CelebA/train/images/ \
    --annotations_dir=/home/dan/datasets/CelebA/train/annotations/ \
    --output=data/train_shards/ \
    --num_shards=500