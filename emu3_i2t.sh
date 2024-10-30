
export CUDA_VISIBLE_DEVICES="2,3"
# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model emu3 \
#     --model_args pretrained="BAAI/Emu3-Chat"\
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix emu3_mmvet \
#     --output_path ./logs/


for i in 4 8; do
    echo "Running with ${i}bit"
    python3 -m accelerate.commands.launch \
        --num_processes=1 \
        -m lmms_eval \
        --model emu3 \
        --model_args pretrained="BAAI/Emu3-Chat",quantized="lmms_eval/models/Emu3/Emu3-Chat-${i}bit"\
        --tasks pope \
        --batch_size 1 \
        --output_path ./logs/${i}bit/pope/
done
