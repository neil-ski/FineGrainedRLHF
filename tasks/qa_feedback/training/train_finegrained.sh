accelerate launch \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 1 \
    --mixed_precision bf16 \
    tasks/qa_feedback/training/train_finegrained.py --config tasks/qa_feedback/training/fine_grained_config.yml