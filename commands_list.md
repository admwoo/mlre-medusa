# Training Commands

## How To Use the Training Commands
--data_path should be the file path of where the ShareGPT dataset is located

--output_dir will denote the name of the directory in which the trained weights are located

--medusa_num_heads denotes the number of heads that the medusa architecture will use 

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

## Extended Medusa Heads Training
```
python medusa/train/train_extension.py \
  --data_path "/home/adamwoo/mlre-medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json" \
  --medusa_layer_config 4 4 4 4 4 \
  --model_name_or_path "lmsys/vicuna-7b-v1.3" \
  --output_dir "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/adamwoo/output_custom/" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-3 \
  --model_max_length 2048 \
  --bf16 True \
  --save_total_limit 3 \
  --resume_from_checkpoint "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/adamwoo/output_custom/_medusa_mlp_vicuna-7b-v1.3_medusa_[4, 4, 4, 4, 4]_lr_0.001/checkpoint-2000"
  
  
```


# Generate Model Answers On MT-Bench

## How to Use Model Generation Commands

You should be located in the llm_judge directory

--model-path should be a file location pointing to the model weights that want to use for generation

Use the gen_model_answer file that corresponds to the architecture you want to generate for

## Baseline Generation
```
python gen_model_answer_baseline_new_2.py --model-path "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/nrgamota/Medusa/output_medusa_1_full_vicuna-7b-v1.3_medusa_5_lr_0.001_layers_1" \
  --model-id medusa-vicuna-7b-v1.3-0
```
This uses the base vicuna model to generate answers

## Medusa Generation
```
python gen_model_answer_medusa_new_2.py --model-path "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/nrgamota/Medusa/output_1_full_medusa_mlp_vicuna-7b-v1.3_medusa_5_lr_0.001_layers_1" \
  --model-id medusa-vicuna-7b-v1.3-0
```

You can use the replication weights posted on Huggingface

```
python gen_model_answer_medusa_new_2.py --model-path Nickg22/Medusa-Replication --model-id medusa-vicuna-7b-v1.3-0
```

## MLP Generation
```
python gen_model_answer_medusa_new_2_mlp.py --model-path "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/nrgamota/Medusa/mlp_256_128_output_medusa_mlp_extn_vicuna-7b-v1.3_medusa_5_lr_0.001_layers_1" \
  --model-id medusa-vicuna-7b-v1.3-0
```

## Extended Medusa Generation
```
python gen_model_answer_extension.py --model-path "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/adamwoo/output_custom/_medusa_mlp_vicuna-7b-v1.3_medusa_[4, 4, 4, 4, 4]_lr_0.001" \
  --model-id medusa-vicuna-7b-v1.3-0
```
## MLP Extensible Generation
```
python gen_model_extend_mlp.py --model-path "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/nrgamota/Medusa/mlp_256_128_output_medusa_mlp_extn_vicuna-7b-v1.3_medusa_5_lr_0.001_layers_1" \
  --model-id medusa-vicuna-7b-v1.3-0
```

Command using weights from Huggingface
```
python gen_model_extend_mlp.py --model-path Nickg22/MLP-Extensible-Extension \
  --model-id medusa-vicuna-7b-v1.3-0
```

# LLM Generate Judgement Command
It is necessary to set a valid API key through OPENAI
```
export OPENAI_API_KEY=$OPENAI_API_KEYs
```

Execute this command in the llm_judge directory

The difference between these judgement commands is that the --model-list parameter needs to correspond to the outputed name of the 
generated answers from the gen_model_answer script


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

Execute this command in the llm_judge directory after generating the judgement

```
python show_result2.py --judge-model gpt-3.5-turbo
```

# Output Metrics of Performance on MT-Bench

Execute this command in the llm_judge directory after running gen_model_answer to create the .jsonl file. 

The quality flags are optional and the values to use are obtained from the show_result2.py script.

```
python calculate_tokens_per_second.py data/mt_bench/model_answer/medusa-vicuna-7b-v1.3-0-temperature-0.0-posterior_threshold-0.09-posterior_alpha-0.3-top_p-0.8-sampling-typical-fast-False.jsonl \
--quality 6.71 \
--quality_per_subject 4.25 7.325 8.7 4.85 4.65 7.76 8.83 7.89
```

# Commands to replicate tree pregeneration

Execute these commands in the medusa/eval directory

## Generate Heads Accuracy
```
python heads_accuracy.py --model_path "/scratch/eecs498f25s006_class_root/eecs498f25s006_class/nrgamota/Medusa/output_1_full_medusa_mlp_vicuna-7b-v1.3_medusa_5_lr_0.001_layers_1" --model_name 'medusa-vicuna-7b-v1.3' --medusa_num_heads 5 --data_path '../../../alpaca/alpaca_eval.json'
```

## Generate Pregenerated Tree

Note that you need to install pygraphviz to utilize this command

```
python gen_results.py --accuracy-path 'medusa-vicuna-7b-v1.3_heads_accuracy.pt' --output-path 'graph.jpg'
```





```