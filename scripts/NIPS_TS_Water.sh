
# python main.py --anormly_ratio 0.6 --num_epochs 5   --batch_size 256 --mode train   --dataset NIPS_TS_Water  --data_path NIPS_TS_GECCO  --channels 9   --win_size 18  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 5 --λ 1.0

python main.py --anormly_ratio 0.6 --num_epochs 5   --batch_size 256 --mode test    --dataset NIPS_TS_Water    --data_path NIPS_TS_GECCO  --channels 9   --win_size 18  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 5 --λ 1.0


