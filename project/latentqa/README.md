# LatentQA

[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-yellow)](https://huggingface.co/collections/Transluce/scalably-extracting-latent-representations-of-users)

This project contains code to train and evaluate LatentQA decoders for user modeling.

See [Scalably Extracting Latent Representations of Users](https://transluce.org/user-modeling) for more details.

## Setup

Follow the instructions in the [Installation section of the README](../../README.md#installation) in the root of the repo. Then install and activate the `latentqa` environment by running
```bash
luce install latentqa
luce activate latentqa
```
This will cd to the latentqa project folder `project/latentqa`, which is where all scripts should be executed from.

## Reproducing Blogpost Plots

To reproduce the plots from the blogpost, we uploaded checkpoints and evaluation results on S3, which can be downloaded with the following command after installing [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html):
```bash
aws s3 cp s3://transluce-public/user-modeling/latentqa.tar.gz .
tar -xzf latentqa.tar.gz
```

Once downloaded, you can generate the plots using the notebook [`notebooks/plots.ipynb`](notebooks/plots.ipynb).

To run the notebook, follow the instructions [here](../../README.md#using-jupyter-notebooks):
```bash
luce nb register latentqa
luce nb start --port <port>
```

## Chatting with a Decoder

To chat with one of the decoders that we trained (uploaded on [HuggingFace](https://huggingface.co/collections/Transluce/scalably-extracting-latent-representations-of-users)), see the demo notebook [`notebooks/decoder_chat.ipynb`](notebooks/decoder_chat.ipynb).

## Training Your Own Decoder

### Define Constants
Replace `HF_MODEL_ID` and `HF_MODEL_NAME` with the appropriate values.
```bash
export DATA_DIR=<...>
export HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
export HF_MODEL_NAME=llama31_8b
export OPENAI_API_KEY=<...>
export ANTHROPIC_API_KEY=<...>
export HF_TOKEN=<...>
export WANDB_API_KEY=<...>
```

### Training and Evaluation Data

#### For Llama-3.1-8B-Instruct
You can download all the training and evaluation data that we used in our experiments for Llama3.1-8B-Instruct with the following command after installing [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html):
```bash
aws s3 cp s3://transluce-public/user-modeling/latentqa.tar.gz .
tar -xzf latentqa.tar.gz
```

#### Generating Data for a New Model
After following the steps in [`project/user_modeling`](../user_modeling/README.md), generate question-answer pairs based on SynthSys for training.

General format (mostly free-form):
```bash
python scripts/generate_lqa_data.py \
    --hf_dataset_id Transluce/SynthSys-Llama-3.1-8B-Instruct \
    --full_data_db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/train_full_data_general_db \
    --question_type general \
    --max_num_insert 300_000
```

Multiple choice format:
```bash
python scripts/generate_lqa_data.py \
    --hf_dataset_id Transluce/SynthSys-Llama-3.1-8B-Instruct \
    --full_data_db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/train_full_data_mc_db \
    --question_type multiple_choice \
    --seed 223 \
    --max_num_insert 300_000
```

Generate paraphrases of direct questions (used as stems for multiple choice evaluation):
```bash
python scripts/generate_mc_stems.py --save_dir $DATA_DIR
```

Construct the training and evaluation datasets:
```bash
python scripts/prepare_lqa_datasets.py \
    --hf_model_id $HF_MODEL_ID \
    --general_db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/train_full_data_general_db \
    --mc_db_dir $DATA_DIR/$HF_MODEL_NAME/synthsys/train_full_data_mc_db \
    --mc_stems_path $DATA_DIR/mc_stems_by_attr_class.pkl \
    --save_dir $DATA_DIR/$HF_MODEL_NAME/
```

### Launch Training
See config file example in `configs/llama31_8b.py`.
```bash
python scripts/launcher.py --config_file configs/llama31_8b.py -f scripts/train_lqa.py -g 1
```

We uploaded the 15→0 and all→all checkpoints trained on SynthSys attributes for Llama3.1-8B-Instruct on [HuggingFace](https://huggingface.co/collections/Transluce/scalably-extracting-latent-representations-of-users).

### Launch Evaluations
For reading evaluations:
```bash
python scripts/launcher.py --config_file configs/llama31_8b.py -f scripts/run_reading_eval.py -g 1
```

For steering evaluations:
```bash
python scripts/launcher.py --config_file configs/llama31_8b.py -f scripts/run_steering_eval.py -g 1
```

For circuit-based steering evaluations:
```bash
python scripts/launcher.py --config_file configs/llama31_8b.py -f scripts/run_circuit_steering_eval.py -g 1
```
