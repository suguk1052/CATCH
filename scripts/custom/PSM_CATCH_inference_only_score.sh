python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "PSM.csv" \
  --model-name "catch.CATCH" \
  --model-hyper-params '{"pretrained_model":"result/checkpoints/CATCH/PSM.pt"}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --save-path "score/CATCH_inference_only"
