

#Accuracy : 0.9930, Precision : 0.9211, Recall : 0.9080, F-score : 0.9145

# python main.py --anormly_ratio 0.234 --num_epochs 1   --batch_size 32 --mode train   --dataset SMD  --data_path SMD   --channels 38   --win_size 65   --jitter_scale_ratio 0.2 --jitter_ratio 0 --max_seg 5 --λ 0.05

python main.py --anormly_ratio 0.234 --num_epochs 1   --batch_size 32 --mode test   --dataset SMD  --data_path SMD   --channels 38   --win_size 65   --jitter_scale_ratio 0.2 --jitter_ratio 0 --max_seg 5 --λ 0.05

