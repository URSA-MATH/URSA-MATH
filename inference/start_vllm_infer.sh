TEMPERATURE=0.2
DATASET="mathvista" # dynamath, wemath, mathvista, mathverse, mathvision
IMAGE_ROOT=""
GENERATE_NUM=1

OUTPUT_FILE="./mathvista_$GENERATE_NUM.jsonl"
DATA_PATH="./data/mathvista/mathvista_testmini.jsonl"
MODEL_PATH="./URSA-7B"

echo "Running inference on data_path: $DATA_PATH"
echo "Save output at $OUTPUT_FILE"

CUDA_VISIBLE_DEVICES=0 python3 inference/vllm_infer.py \
  --model $MODEL_PATH \
  --dataset $DATASET \
  --temperature $TEMPERATURE \
  --data_path $DATA_PATH \
  --output_file $OUTPUT_FILE \
  --image_root $IMAGE_ROOT \
  --num_return_sequences $GENERATE_NUM \

