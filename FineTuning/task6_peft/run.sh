#!/bin/bash
for method in ia3 prompt_tuning lora; do
    python run_train.py --method "$method"
done
