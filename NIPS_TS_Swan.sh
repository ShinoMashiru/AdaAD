python main.py --anormly_ratio 0.4 --num_epochs 5   --batch_size 256 --mode train   --dataset NIPS_TS_Swan  --data_path NIPS_TS_Swan   --channels 38   --win_size 18  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 2 --λ 1.0


python main.py --anormly_ratio 3.0 --num_epochs 5   --batch_size 256 --mode test    --dataset NIPS_TS_Swan  --data_path NIPS_TS_Swan   --channels 38   --win_size 18  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 2 --λ 1.0

