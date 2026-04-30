# CATCH 추론 전용 실행 (학습 없이)

이 모드는 기존 학습+추론 파이프라인에서 **학습을 생략**하고, 저장된 CATCH 체크포인트를 로드해 **추론 + metric 계산 + 새 레코드 저장**만 수행합니다.

## 1) 먼저 1회 학습 실행 시 체크포인트 저장

학습 실행 시 `model_hyper_params`에 `checkpoint_path`를 넣으면 최적 가중치와 scaler가 저장됩니다.

예시:
```bash
python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_label_multi_config.json" \
  --data-name-list "PSM.csv" \
  --model-name "catch.CATCH" \
  --model-hyper-params '{"checkpoint_path":"result/checkpoints/CATCH/PSM.pt"}' \
  --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/CATCH_train"
```

## 2) 추론 전용 실행

```bash
bash scripts/custom/PSM_CATCH_inference_only_label.sh
```

- 출력 레코드: `result/label/CATCH_inference_only/*.csv.tar.gz`
- 레코드에 `inference_time`, `inference_speed`(samples/sec)가 포함되어 추론 속도 비교 가능

## 3) score 버전 추론 전용 실행

```bash
bash scripts/custom/PSM_CATCH_inference_only_score.sh
```

- 출력 레코드: `result/score/CATCH_inference_only/*.csv.tar.gz`
