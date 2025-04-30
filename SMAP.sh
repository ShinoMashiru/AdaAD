
python main.py --anormly_ratio 3.0 --num_epochs 5   --batch_size 256 --mode train --dataset SMAP  --data_path SMAP   --channels 25  --win_size  150   --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 7 --λ 0.5

python main.py --anormly_ratio 3.0 --num_epochs 5   --batch_size 256 --mode test  --dataset SMAP  --data_path SMAP   --channels 25   --win_size 150   --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 7 --λ 0.5


