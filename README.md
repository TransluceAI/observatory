# Transluce

Transluce is an independent research lab building open, scalable technology for understanding AI systems and steering them in the public interest.

This repository hosts code for the following projects:

- [User Modeling](https://transluce.org/user-modeling), which is about creating datasets to study language model's latent user representations;
- [LatentQA](https://transluce.org/user-modeling), which trains decoders to read and steer user representations in language models;
- [Neuron Descriptions](https://transluce.org/neuron-descriptions), which automatically generates high-quality descriptions of language model neurons;
- The [Monitor](https://transluce.org/observability-interface) interface, which helps humans observe, understand, and steer the internal computations of language models.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
  - [User Modeling](#user-modeling)
  - [LatentQA](#latentqa)
  - [Neuron Descriptions](#neuron-descriptions)
  - [Monitor](#monitor)
- [Using `luce`](#using-luce)
  - [Package and environment management](#package-and-environment-management)
  - [Using Jupyter notebooks](#using-jupyter-notebooks)
- [Citation](#citation)

## Installation

First clone this repo:

```bash
git clone https://github.com/TransluceAI/observatory.git
```

### Installing `luce`

Next, we'll install `luce`, a command-line tool that manages project environments and dependencies. It will significantly simplify setup for downstream projects.

To install `luce`, add the following to your shell profile (e.g., `.bashrc`, `.zshrc`):

```bash
# ... existing shell config
export TRANSLUCE_HOME=<absolute_path_to_repo>
source $TRANSLUCE_HOME/lib/lucepkg/scripts/shellenv.sh
```

Make sure to `source` your shell profile:

```bash
source ~/.bashrc  # for bash users
# OR
source ~/.zshrc   # for zsh users
```

Then run:

```bash
luce uv install  # install uv package manager
luce install     # install base environment
```

to install the `uv` package manager and base virtual environment, respectively. The base venv includes basic packages like the Jupyter kernel, pre-commit, etc.

### Setting up environment variables

Finally, clone the `.env.template` file to `.env` and fill in the missing values.

```bash
cp .env.template .env
```

These variables are always required:

- `OPENAI_API_KEY` / `OPENAI_API_ORG`: OpenAI API key and organization ID.
- `ANTHROPIC_API_KEY`: Anthropic API key.
- `HF_TOKEN`: Required for accessing gated models (e.g., Llama-3.1) on HuggingFace.

The rest of the variables are only required for running the NeuronDB or Monitor; you can safely ignore them for now.

## Getting Started

### User Modeling
See the [user modeling README](project/user_modeling/README.md) for generating data to study how language models represent users.

### LatentQA
See the [latentqa README](project/latentqa/README.md) for training LatentQA decoders to read and steer user representations in language models.

### Neuron Descriptions

See the [description generation README](project/expgen/README.md) for generating neuron descriptions automatically.

### Monitor

See the [Monitor README](project/monitor/README.md) for instructions on how to set up a local development environment.

## Using `luce`

### Package and environment management

Each folder under `lib/` and `project/` has its own venv. Use `luce` to:

```bash
# Install all dependencies
luce install --all

# Install and activate a specific package
luce install <package_name>

# Activate a venv and cd to its directory
luce activate <package_name>

# Deactivate the current package
deactivate
```

You may need to use `--force` to reinstall a package that already exists; this removes `poetry.lock`:

```bash
luce install --force <package_name>
```

### Using Jupyter notebooks

We include utilities to register Jupyter kernels into the top-level environment. To register a kernel for a package, run:

```bash
luce nb register <package_name>
```

To start a notebook server that can call any of the registered kernels, run:

```bash
luce nb start --port <port>
```

You'll get a readout of the notebook server URL, which you can use to connect to the notebook server via the web or an IDE.

## Support

If you run into any issues, please file an issue or reach out to us at [info@transluce.org](mailto:info@transluce.org). We're happy to help!

## Citation

If you use this code for your research, please cite our work:

```bibtex
@misc{choi2025scalably,
  author       = {Choi, Dami and Huang, Vincent and Schwettmann, Sarah and Steinhardt, Jacob},
  title        = {Scalably Extracting Latent Representations of Users},
  year         = {2025},
  month        = {November},
  day          = {25},
  howpublished = {\url{https://transluce.org/user-modeling}}
}
```

```bibtex
@misc{choi2024automatic,
  author       = {Choi, Dami and Huang, Vincent and Meng, Kevin and Johnson, Daniel D and Steinhardt, Jacob and Schwettmann, Sarah},
  title        = {Scaling Automatic Neuron Description},
  year         = {2024},
  month        = {October},
  day          = {23},
  howpublished = {\url{https://transluce.org/neuron-descriptions}}
}
```
