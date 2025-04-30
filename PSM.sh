
python main.py --anormly_ratio 0.25 --num_epochs 2    --batch_size 128  --mode train --dataset PSM  --data_path PSM     --channels 25  --loss_fuc MSE  --win_size 60  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 5 --λ 0.2

python main.py --anormly_ratio 0.25 --num_epochs 2    --batch_size 128  --mode test  --dataset PSM  --data_path PSM     --channels 25  --loss_fuc MSE  --win_size 60  --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 5 --λ 0.2