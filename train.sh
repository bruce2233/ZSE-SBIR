 #!/bin/bash
python -u train.py --data_path ./datasets \
                   --dataset sketchy_extend \
                   --test_class test_class_sketchy25 \
                   --batch 5 \
                   --epoch 10 \
                   -s ./checkpoints/sketchy_ext \
                   -c 0 \
                   -r rn \
                   --valid_shrink_sk 200 \
                   --valid_shrink_im 100 \
                   --pretrain_load ./checkpoint/sketchy_ext/best_checkpoint.pth \
                   --test_sk 5 \
                   --test_im 5
