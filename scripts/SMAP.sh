#Accuracy : 0.9936, Precision : 0.9712, Recall : 0.9806, F-score : 0.9759

# python main.py --anormly_ratio 0.47 --num_epochs 1   --batch_size 512 --mode train --dataset SMAP  --data_path SMAP   --channels 25  --win_size  170   --jitter_scale_ratio 1.4 --jitter_ratio 0.9 --max_seg 6 --λ 0.0

python main.py --anormly_ratio 0.47 --num_epochs 1   --batch_size 512 --mode test  --dataset SMAP  --data_path SMAP   --channels 25   --win_size 170   --jitter_scale_ratio 1.4 --jitter_ratio 0.9 --max_seg 6 --λ 0.0
