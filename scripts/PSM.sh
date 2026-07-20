#Accuracy : 0.9939, Precision : 0.9905, Recall : 0.9869, F-score : 0.9887 

# python main.py --anormly_ratio 0.30 --num_epochs 1    --batch_size 256  --mode train --dataset PSM  --data_path PSM     --channels 25  --loss_fuc MSE  --win_size 170  --jitter_scale_ratio 1.1 --jitter_ratio 0.2 --max_seg 2 --λ 0.1

python main.py --anormly_ratio 0.30 --num_epochs 1    --batch_size 256  --mode test --dataset PSM  --data_path PSM     --channels 25  --loss_fuc MSE  --win_size 170 --jitter_scale_ratio 1.1 --jitter_ratio 0.2 --max_seg 2 --λ 0.1


