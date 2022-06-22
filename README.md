**YaLM 100B** is a GPT-like neural network for generating and processing text. It can be used freely by developers and researchers from all over the world.

The model leverages 100 billion parameters. It took 65 days to train the model on a cluster of 800 A100 graphics cards and 1.7 TB of online texts, books, and countless other sources in both English and Russian.

Training details and best practices on acceleration and stabilizations can be found on **[Medium](https://medium.com/p/d1df53d0e9a6)** (English) and **[Habr](https://habr.com/ru/company/yandex/blog/672396/)** (Russian) articles.

# Setup

Make sure to have 200GB of free disk space before downloading weights. The model *(code is based on [microsoft/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3](https://github.com/microsoft/DeepSpeedExamples/tree/068e6561188e9192104e014f70fbe25224b5eb62/Megatron-LM-v1.1.5-ZeRO3))* is supposed to run on multiple GPUs with tensor parallelism. It was tested on 4 (A100 80g) and 8 (V100 32g) GPUs, but is able to work with different configurations with ≈200GB of GPU memory in total which divide weight dimensions correctly (e.g. 16, 64, 128).

## Downloading checkpoint

* Run `bash download/download.sh` to download model weights and vocabulary.
* By default, weights will be downloaded to `./yalm100b_checkpoint/weights/`, and vocabulary will be downloaded to `./yalm100b_checkpoint/vocab/`.

## Docker

* We [published](https://hub.docker.com/r/yandex/yalm-cuda11-ds) image on Docker Hub, it can be pulled with `docker/pull.sh`. It is compatible with A100 and V100.
* Alternatively, you can build docker image from source using `docker/build.sh` (which will just build docker image from `docker/Dockerfile`).
* To run container, use `docker/run.sh` *(volumes, name and other parameters can be changed)*.

# Usage

You can start with the following scripts:
* `examples/generate_interactive.sh`: interactive generation from command line, the simplest way to try the model.
* `examples/generate_conditional_sampling.sh`: conditional generation with sampling strategy. Top-p is used by default, feel free to change temperature or use top-k. Input is jsonlines (example: `examples/example_cond_input.json`), output will be the same jsonlines with generated text field added to each line.
* `examples/generate_conditional_greedy.sh`: same as previous, but generation is greedy. Suitable for solving problems with few-shot.
* `examples/generate_unconditional.sh`: unconditional generation. No input is used, output will be jsonlines.

# License

The model is published under the Apache 2.0 license that permits both research and commercial use, Megatron-LM is licensed under the [Megatron-LM license](megatron_lm/LICENSE).
