# User Modeling

[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-yellow)](https://huggingface.co/collections/Transluce/scalably-extracting-latent-representations-of-users)

This project contains code to generate the core data for studying user modeling.

See [Scalably Extracting Latent Representations of Users](https://transluce.org/user-modeling) for more details.

Note that the datasets for Llama3.1-8B-Instruct and Llama3.1-70B-Instruct are already on HuggingFace (see the [collection](https://huggingface.co/collections/Transluce/scalably-extracting-latent-representations-of-users)). The commands below can reproduce them, but are primarily for if you want to run belief evaluation for new models.

## Setup

Follow the instructions in the [Installation section of the README](../../README.md#installation) in the root of the repo. Then install and activate the `user_modeling` environment by running
```bash
luce install user_modeling
luce activate user_modeling
```
This will cd to the user_modeling project folder `project/user_modeling`, which is where all scripts should be executed from.

## Define Constants
Replace `HF_MODEL_ID` and `HF_MODEL_NAME` with the appropriate values.
```bash
export DATA_DIR=<...>
export PORT=2733
export HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
export HF_MODEL_NAME=llama31_8b
export OPENAI_API_KEY=<...>
export ANTHROPIC_API_KEY=<...>
export HF_TOKEN=<...>
```

## SynthSys

### Generate System and User Prompts
Generate user prompts using o3:
```bash
python scripts/generate_user_prompts.py \
    --db_dir $DATA_DIR/user_prompts_db \
    --min_num_prompts 100 \
    --num_candidates_per_call 100 \
    --generation_provider openai \
    --generation_model_name o3 \
    --generation_max_new_tokens 10_000
```

Generate user prompts using Claude Opus 4:
```bash
python scripts/generate_user_prompts.py \
    --db_dir $DATA_DIR/user_prompts_db \
    --min_num_prompts 100 \
    --num_candidates_per_call 100 \
    --generation_provider anthropic \
    --generation_model_name claude-opus-4-1-20250805 \
    --generation_max_new_tokens 5_000
```

Generate system prompts using o3:
```bash
python scripts/generate_system_prompts.py \
    --db_dir $DATA_DIR/system_prompts_db \
    --min_num_prompts 100 \
    --num_candidates_per_call 100 \
    --provider openai \
    --model_name o3 \
    --max_new_tokens 10_000
```

Generate system prompts using Claude Opus 4:
```bash
python scripts/generate_system_prompts.py \
    --db_dir $DATA_DIR/system_prompts_db \
    --min_num_prompts 100 \
    --num_candidates_per_call 100 \
    --provider anthropic \
    --model_name claude-opus-4-1-20250805 \
    --max_new_tokens 5_000
```

### 1. Launch Faithfulness (Belief) Evaluation
Start vLLM server for generating from the subject model:
```bash
vllm serve $HF_MODEL_ID \
    --tensor-parallel-size <number of gpus> \
    --host 0.0.0.0 \
    --port $PORT
```

For the train split:
```bash
python scripts/run_synthsys_faithfulness_eval.py \
    --hf_model_id $HF_MODEL_ID \
    --num_subject_completions 2 \
    --db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/train_faith_db \
    --system_prompt_db_dir $DATA_DIR/system_prompts_db \
    --user_prompt_db_dir $DATA_DIR/user_prompts_db \
    --split train \
    --subject_gen_client_url http://0.0.0.0:$PORT/v1 \
    --num_examples 1_000_000
```

For the test split:
```bash
python scripts/run_synthsys_faithfulness_eval.py \
    --hf_model_id $HF_MODEL_ID \
    --num_subject_completions 2 \
    --db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/test_faith_db \
    --system_prompt_db_dir $DATA_DIR/system_prompts_db \
    --user_prompt_db_dir $DATA_DIR/user_prompts_db \
    --user_prompt_type user_prompt \
    --split test \
    --subject_gen_client_url http://0.0.0.0:$PORT/v1 \
    --num_examples 10_000
```

### 2. Evaluate Baseline Beliefs (No System Prompt)
Run this if you want to evaluate all available user prompts in `$DATA_DIR/user_prompts_db`:
```bash
python scripts/run_synthsys_faithfulness_eval.py \
    --hf_model_id $HF_MODEL_ID \
    --num_subject_completions 2 \
    --db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/baseline_faith_db \
    --user_prompt_db_dir $DATA_DIR/user_prompts_db \
    --user_prompt_type user_prompt \
    --subject_gen_client_url http://0.0.0.0:$PORT/v1 \
    --eval_baseline True
```

If you only want to evaluate user prompts from train/test faithfulness eval:
```bash
python scripts/run_synthsys_faithfulness_eval.py \
    --hf_model_id $HF_MODEL_ID \
    --num_subject_completions 2 \
    --db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/baseline_faith_db \
    --user_prompt_db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/<train_faith_db or test_faith_db> \
    --user_prompt_type faith_result \
    --subject_gen_client_url http://0.0.0.0:$PORT/v1 \
    --eval_baseline True
```

### 3. Aggregate Faithfulness Eval Results

For the train split:
```bash
python scripts/run_subject_input_aggregation.py \
    --db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/train_subject_input_db \
    --faithfulness_eval_db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/train_faith_db \
    --baseline_faithfulness_eval_db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/baseline_faith_db
```

For the test split:
```bash
python scripts/run_subject_input_aggregation.py \
    --db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/test_subject_input_db \
    --faithfulness_eval_db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/test_faith_db \
    --baseline_faithfulness_eval_db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/baseline_faith_db
```

### 4. Filter to Create the Final SynthSys Dataset

For the train split:
```bash
python scripts/create_synthsys.py \
    --db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/train_subject_input_db \
    --save_path $DATA_DIR/$HF_MODEL_NAME/synthsys/train_filtered_data.pkl
```

For the test split:
```bash
python scripts/create_synthsys.py \
    --db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/test_subject_input_db \
    --save_path $DATA_DIR/$HF_MODEL_NAME/synthsys/test_filtered_data.pkl
```

## SelfDescribe

### 1. Download Self-Descriptions
```bash
aws s3 cp s3://transluce-public/user-modeling/wikipedia_stereotypes.jsonl $DATA_DIR
```

### 2. Launch Belief Evaluation and Create the SelfDescribe(8B) Dataset
```bash
python scripts/create_self_describe.py \
    --stereotypes_path $DATA_DIR/wikipedia_stereotypes.jsonl \
    --subject_hf_model_id $HF_MODEL_ID \
    --save_dir $DATA_DIR/$HF_MODEL_NAME/self_describe \
    --subject_client_url http://0.0.0.0:$PORT/v1
```

## PRISM

### Launch Belief Evaluation and Create the PRISM(8B, Gender) Dataset
```bash
python scripts/create_prism.py \
    --save_dir $DATA_DIR/$HF_MODEL_NAME/prism \
    --subject_client_url http://0.0.0.0:$PORT/v1
```
