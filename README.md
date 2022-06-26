# YaLM 100B
**YaLM 100B** is a GPT-like neural network for generating and processing text. It can be used freely by developers and researchers from all over the world.

The model leverages 100 billion parameters. It took 65 days to train the model on a cluster of 800 A100 graphics cards and 1.7 TB of online texts, books, and countless other sources in both English and Russian.

Training details and best practices on acceleration and stabilizations can be found on **[Medium](https://medium.com/p/d1df53d0e9a6)** (English) and **[Habr](https://habr.com/ru/company/yandex/blog/672396/)** (Russian) articles.

We used DeepSpeed to train the model and drew inspiration from Megatron-LM example. However, the code in this repo is not the same code that was used to train the model. Rather it is stock example from DeepSpeed repo with minimal changes needed to infer our model.

## Setup

Make sure to have 200GB of free disk space before downloading weights. The model *(code is based on [microsoft/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3](https://github.com/microsoft/DeepSpeedExamples/tree/068e6561188e9192104e014f70fbe25224b5eb62/Megatron-LM-v1.1.5-ZeRO3))* is supposed to run on multiple GPUs with tensor parallelism. It was tested on 4 (A100 80g) and 8 (V100 32g) GPUs, but is able to work with different configurations with ≈200GB of GPU memory in total which divide weight dimensions correctly (e.g. 16, 64, 128).

### Downloading checkpoint

* Run `bash download/download.sh` to download model weights and vocabulary.
* By default, weights will be downloaded to `./yalm100b_checkpoint/weights/`, and vocabulary will be downloaded to `./yalm100b_checkpoint/vocab/`.

### Docker

* We [published](https://hub.docker.com/r/yandex/yalm-cuda11-ds) image on Docker Hub, it can be pulled with `docker/pull.sh`. It is compatible with A100 and V100.
* Alternatively, you can build docker image from source using `docker/build.sh` (which will just build docker image from `docker/Dockerfile`).
* To run container, use `docker/run.sh` *(volumes, name and other parameters can be changed)*.

## Usage

You can start with the following scripts:
* `examples/generate_interactive.sh`: interactive generation from command line, the simplest way to try the model.
* `examples/generate_conditional_sampling.sh`: conditional generation with sampling strategy. Top-p is used by default, feel free to change temperature or use top-k. Input is jsonlines (example: `examples/example_cond_input.json`), output will be the same jsonlines with generated text field added to each line.
* `examples/generate_conditional_greedy.sh`: same as previous, but generation is greedy. Suitable for solving problems with few-shot.
* `examples/generate_unconditional.sh`: unconditional generation. No input is used, output will be jsonlines.

## License

The model is published under the Apache 2.0 license that permits both research and commercial use, Megatron-LM is licensed under the [Megatron-LM license](megatron_lm/LICENSE).

## Training details

### Dataset composition

Dataset used for the training of YaLM-100B is comprised of the following parts (rough percentages are measured in tokens seen by the model)

* **25%** [The Pile](https://pile.eleuther.ai/) — open English dataset by Eleuther AI team

* **75%** Texts in Russian collected by our team (percentages  of the whole dataset are given) 

    * 49% Russian web pages from Yandex Search index filtered from ~100Tb to ~1Tb by the following heuristics:
      1. LSH Deduplication — clusters of similar texts were truncated to just one text each
      2. Length filtration — too short or too long texts or texts with too few natural sentences were discarded.
      3. Entropy filtration — texts with too high or too low entropy were discarded
      4. Domain filtration — domains with repetitive texts (like online retail) were discarded
      5. Classifier filtration — dataset of good texts was collected in a manner similar to WebText from pages linked in tweets in Russian that have at least one reply. Then a classifier was trained to distinguish those good texts from random pages from the dataset. Texts from the original crawled dataset with low classifier scores were then discarded
    
    * 12% News from various sources from Yandex Search index
    
    * 10% Books from the dataset used in [Russian Distributional Thesarus](https://russe.nlpub.org/downloads/)
    
    * 3% Misc texts from the [Taiga Dataset](https://tatianashavrina.github.io/taiga_site/)
    
    * 1.5% Dialogues from social media preprocessed in a manner similar to how Reddit is proccessed in The Pile
    
    * 0.5% Russian portion of Wikipedia

Some subsets were traversed up to 3 times during the training.


### Training process

Model was trained on a cluster of 800 A100 for ~65 days. In that time it consumed 300B tokens. You can see TensorBoard with LR and ramp up schedule, training metrics and our "thermometers" on the [HF page](https://huggingface.co/yandex/yalm-100b).
