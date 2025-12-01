

##cd /e/newpapercode

##conda activate exp_pytorch

##source /d/anaconda/etc/profile.d/conda.sh

##python main.py --anormly_ratio 1 --num_epochs 15   --batch_size 256 --mode train   --dataset MSL  --data_path MSL   --channels 55   --win_size 150  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 7 --λ 0.0


##python main.py --anormly_ratio 0.4 --num_epochs 2   --batch_size 256 --mode test   --dataset MSL  --data_path MSL   --channels 55   --win_size 150  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 7 --λ 0.0

#python main.py --anormly_ratio 0.4 --num_epochs 2   --batch_size 256 --mode train   --dataset MSL  --data_path MSL   --channels 55   --win_size 150  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 7 --λ 0.0

#python main.py --anormly_ratio 0.74 --num_epochs 2   --batch_size 256 --mode test   --dataset MSL  --data_path MSL   --channels 55   --win_size 150  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 7 --λ 0.0

python main.py --anormly_ratio 0.4 --num_epochs 5   --batch_size 256 --mode train   --dataset MSL  --data_path MSL   --channels 55   --win_size 150  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 10 --λ 0.0

python main.py --anormly_ratio 0.63 --num_epochs 5   --batch_size 256 --mode test   --dataset MSL  --data_path MSL   --channels 55   --win_size 150  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 10 --λ 0.0










































