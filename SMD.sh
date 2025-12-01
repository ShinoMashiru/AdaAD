#python main.py --anormly_ratio 0.1 --num_epochs 2   --batch_size 256 --mode train  --dataset SMD  --data_path SMD   --channels 38   --win_size 150   --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 3 --λ 0.05
#python main.py --anormly_ratio 0.4 --num_epochs 2   --batch_size 256 --mode test  --dataset SMD  --data_path SMD   --channels 38   --win_size 150   --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 3 --λ 0.05

#!/bin/bash
# 仅清理 Python 进程占用的 GPU 显存

echo "正在检查 GPU 占用情况..."
nvidia-smi

# 找到所有占用 GPU 的进程 PID 和名称
PROCESSES=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)

if [ -z "$PROCESSES" ]; then
    echo "没有发现占用 GPU 的进程，显存已经空闲。"
else
    echo "发现以下进程正在占用 GPU："
    echo "$PROCESSES"
    
    # 仅筛选 python 相关进程
    PIDS=$(echo "$PROCESSES" | grep python | awk -F ',' '{print $1}' | tr -d ' ')

    if [ -z "$PIDS" ]; then
        echo "没有发现 Python 进程占用 GPU。"
    else
        echo "准备终止以下 Python 进程以释放显存: $PIDS"
        for PID in $PIDS; do
            echo "终止进程 PID: $PID"
            kill -9 $PID
        done
    fi

    echo "显存清理完成。"
    nvidia-smi
fi
#python main.py --anormly_ratio 0.1 --num_epochs 1   --batch_size 256 --mode train  --dataset SMD  --data_path SMD   --channels 38   --win_size 180   --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 3 --λ 0.05
#python main.py --anormly_ratio 0.45 --num_epochs 2   --batch_size 256 --mode test  --dataset SMD  --data_path SMD   --channels 38   --win_size 180   --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 3 --λ 0.05

python main.py --anormly_ratio 0.1 --num_epochs 1   --batch_size 256 --mode train  --dataset SMD  --data_path SMD   --channels 38   --win_size 170   --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 3 --λ 0.05
python main.py --anormly_ratio 0.35 --num_epochs 2   --batch_size 256 --mode test  --dataset SMD  --data_path SMD   --channels 38   --win_size 170   --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 3 --λ 0.05
python main.py --anormly_ratio 0.4 --num_epochs 2   --batch_size 256 --mode test  --dataset SMD  --data_path SMD   --channels 38   --win_size 170   --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 3 --λ 0.05
python main.py --anormly_ratio 0.45 --num_epochs 2   --batch_size 256 --mode test  --dataset SMD  --data_path SMD   --channels 38   --win_size 170   --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 3 --λ 0.05
python main.py --anormly_ratio 0.5 --num_epochs 2   --batch_size 256 --mode test  --dataset SMD  --data_path SMD   --channels 38   --win_size 170   --jitter_scale_ratio 1.1 --jitter_ratio 0.9 --max_seg 3 --λ 0.05