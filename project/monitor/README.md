# Monitor

[Monitor](https://transluce.org/observability-interface) is an AI-driven observability interface designed to help humans observe, understand, and steer the internal computations of language models. This README explains how to deploy a local version of Monitor; the deployed version is available at [https://monitor.transluce.org](https://monitor.transluce.org).

We recommend developing on a GPU node, since the backend server runs language model inference.

## Table of Contents

- [Developing Locally](#developing-locally)
  - [1. NeuronDB](#1-neurondb)
  - [2. Backend ASGI server](#2-backend-asgi-server)
  - [3. Frontend web server](#3-frontend-web-server)

## Developing Locally

First install and activate the `monitor` venv, and `cd` to the `monitor` directory:

```bash
luce install monitor  # omit if you already have the venv installed
luce activate monitor  # activates venv and cds automatically
```

We then need to setup three key components: the NeuronDB database, the backend ASGI server, and the frontend web server.

### 1. NeuronDB

Monitor requires a PostgreSQL database to read neuron descriptions from. We offer Docker images for starting a local Postgres instance. (Don't worry, you don't have to know how Docker works!)

First, if you don't already have Docker installed, see the [Docker installation guide](https://docs.docker.com/get-docker/).

Once that's done, your options are to:

1. **[RECOMMENDED]** Use our [prebuilt Postgres + `pgvector` Docker image](https://hub.docker.com/r/mengkluce/neurondb-llama-3.1-8b-instruct) that contains pre-loaded neuron descriptions for `llama-3.1-8b-instruct`. To use this option, run:

   ```bash
   ./scripts/neurondb.sh  # May need to use `sudo` on Linux
   ```

   to pull the image and start a Postgres container. Note that this image is large (~36GB), so pulling it may take a few dozen minutes on a typical home WiFi connection, or <10m on a datacenter-grade ethernet connection.

2. Use a [fresh unpopulated image](https://hub.docker.com/r/mengkluce/neurondb-fresh) with no pre-loaded neuron descriptions. To use this option, run:

   ```bash
   ./scripts/neurondb.sh --fresh
   ```

   This image is fast to pull and start, but you won't see any neuron descriptions, linter outputs, or steering capabilities in the UI.

The database automatically starts when you run the script; you can check that it's running with `docker ps`.

For the [`DBManager`](../../lib/neurondb/postgres.py) to connect to the database, you'll also need to set the following environment variables in the root `.env` (see [the env section of the main README](../..//README.md#setting-up-environment-variables) if you haven't configured a `.env` yet).

```
PG_USER=clarity
PG_PASSWORD=sysadmin
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=neurons
```

If you want to learn more about the NeuronDB and/or Postgres setup, see the [NeuronDB README](../../lib/neurondb/README.md).

### 2. Backend ASGI server

First download the Llama-3.1 8B Instruct model to your local cache (`HF_TOKEN` must be set in your root `.env`):

```bash
python3 scripts/download_llm.py
```

Then run the API server (`-d` for dev mode):

```bash
./scripts/api.sh  -d
```

### 3. Frontend web server

First make sure Node.js is installed; if not, run:

```bash
luce node install
```

Then, run the web server; this script also installs dependencies in `web/` if `web/node_modules/` doesn't exist:

```bash
./scripts/web.sh
```

You can then view the web app at [`http://localhost:3000`](http://localhost:3000).

## Deploying to Modal

If you don't already have a Modal account, first [create one](https://modal.com). Then run the following command to authenticate:

```bash
modal token new
```

Next, open [`monitor/server_modal.py`](./monitor/server_modal.py) and see if you'd like to change any of the default settings, e.g., the GPU count/type, concurrency limit, or warm pool.

Finally, deploy by running:

```bash
modal deploy monitor/server_modal.py
```
