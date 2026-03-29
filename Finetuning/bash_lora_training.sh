#!/bin/bash

mkdir -p logs
exec 1> logs/train.log
exec 2> logs/train.err

echo "Strating training"
python3 lora_training.py \
  --model_id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --dataset_name Codatta/MM-Food-100K \
  --output_dir ./output \
  --epochs 2 \
  --batch_size 1 \
  --lr 1e-4

# Check if the python command was successful
if [ $? -eq 0 ]; then
    echo "Finished training successfully!"
else

    echo "ERROR: Training failed! Check logs/train.err for details." >&2
    exit 1
fi