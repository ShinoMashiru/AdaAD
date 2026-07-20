#Accuracy : 0.9967, Precision : 0.9779, Recall : 0.9950, F-score : 0.9863

# python main.py --anormly_ratio 0.31 --num_epochs 1   --batch_size 64 --mode train   --dataset SWAT  --data_path SWAT   --channels 51   --win_size 120   --jitter_scale_ratio 0 --jitter_ratio 4.7 --max_seg 4 --λ 0.01

python main.py --anormly_ratio 0.311 --num_epochs 1   --batch_size 256 --mode test   --dataset SWAT  --data_path SWAT   --channels 51   --win_size 120   --jitter_scale_ratio 0 --jitter_ratio 4.7 --max_seg 4 --λ 0.01
