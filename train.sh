python train.py --data-path D:\\Files\\GitHub\\AdvSemiSeg-Paddle\\dataset\\VOC2012 \
                --restore-from D:\\Files\\GitHub\\AdvSemiSeg-Paddle\\pdparams\\resnet101COCO-41f33a49.pdparams \
                --snapshot-dir snapshots \
                --partial-data 0.125 \
                --num-steps 20000 \
                --lambda-adv-pred 0.01 \
                --lambda-semi 0.1 --semi-start 5000 --mask-T 0.2