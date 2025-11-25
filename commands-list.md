# Training Commands

## Regular Training
```
python -m medusa.train.train_legacy \
  --model_name_or_path lmsys/vicuna-7b-v1.3 \
  --data_path "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/nrgamota/Medusa/ShareGPT/ShareGPT_V4.3_unfiltered_cleaned_split.json" \
  --output_dir "./output_1_full" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-3 \
  --model_max_length 2048 \
  --bf16 True \
  --medusa_num_heads 5 \
  --medusa_num_layers 1
```

## MLP Training
```
python -m medusa.train.train_legacy_mlp \
  --model_name_or_path lmsys/vicuna-7b-v1.3 \
  --data_path "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/nrgamota/Medusa/ShareGPT/ShareGPT_V4.3_unfiltered_cleaned_split.json" \
  --output_dir "./mlp_512_256_output" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-3 \
  --model_max_length 2048 \
  --bf16 True \
  --medusa_num_heads 5 \
  --medusa_num_layers 1
```


# Generate Model Answers On MT-Bench

## Baseline Generation
```
python gen_model_answer_baseline_new_2.py --model-path "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/nrgamota/Medusa/output_medusa_1_full_vicuna-7b-v1.3_medusa_5_lr_0.001_layers_1" \
  --model-id medusa-vicuna-7b-v1.3-0
```

## Medusa Generation
```
python gen_model_answer_medusa_new_2.py --model-path "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/nrgamota/Medusa/output_1_full_medusa_mlp_vicuna-7b-v1.3_medusa_5_lr_0.001_layers_1" \
  --model-id medusa-vicuna-7b-v1.3-0
```

## MLP Generation
```
python gen_model_answer_medusa_new_2_mlp.py --model-path "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/nrgamota/Medusa/mlp_256_128_output_medusa_mlp_extn_vicuna-7b-v1.3_medusa_5_lr_0.001_layers_1" \
  --model-id medusa-vicuna-7b-v1.3-0
```

# LLM Generate Judgement Command
It is necessary to set a valid API key through OPENAI
```
export OPENAI_API_KEY=$OPENAI_API_KEYs
```
## Baseline Judgement
```
python gen_judgement.py --model-list medusa-vicuna-7b-v1.3-0-temperature-0.0-posterior_threshold-0.09-posterior_alpha-0.3-top_p-0.8-sampling-typical-fast-False-baseline --judge-model gpt-3.5-turbo
```

## Medusa Judgement
```
python gen_judgement.py --model-list medusa-vicuna-7b-v1.3-0-temperature-0.0-posterior_threshold-0.09-posterior_alpha-0.3-top_p-0.8-sampling-typical-fast-False --judge-model gpt-3.5-turbo
```

# Show Judgement Command
Use this to get output of quality results based on the generated judgement
```
python show_result2.py --judge-model gpt-3.5-turbo
```

# Output Metrics of Performance on MT-Bench

```
python calculate_tokens_per_second2.py data/mt_bench/model_answer/medusa-vicuna-7b-v1.3-0-temperature-0.0-posterior_threshold-0.09-posterior_alpha-0.3-top_p-0.8-sampling-typical-fast-False.jsonl \
--quality 6.71 \
--quality_per_subject 4.25 7.325 8.7 4.85 4.65 7.76 8.83 7.89
```

# Commands to replicate tree pregeneration

## Generate Heads Accuracy
```
python heads_accuracy.py --model_path "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/nrgamota/Medusa/output_1_full_medusa_mlp_vicuna-7b-v1.3_medusa_5_lr_0.001_layers_1" --model_name 'medusa-vicuna-7b-v1.3' --medusa_num_heads 5 --data_path '../../../alpaca/alpaca_eval.json'
```

## Generate Pregenerated Tree
```
python gen_results.py --accuracy-path 'medusa-vicuna-7b-v1.3_heads_accuracy.pt' --output-path 'graph.jpg'
```
