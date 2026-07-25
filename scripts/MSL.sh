
#Accuracy : 0.9915, Precision : 0.9684, Recall : 0.9795, F-score : 0.9739 

#python main.py --anormly_ratio 0.63 --num_epochs 5   --batch_size 256 --mode train   --dataset MSL  --data_path MSL   --channels 55   --win_size 150  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 10 --λ 0.0

python main.py --anormly_ratio 0.63 --num_epochs 5   --batch_size 256 --mode test   --dataset MSL  --data_path MSL   --channels 55   --win_size 150  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 10 --λ 0.0
















