#!/bin/bash

mkdir -p yalm100b_checkpoint/vocab yalm100b_checkpoint/weights

cd yalm100b_checkpoint/vocab
curl --remote-name-all https://yalm-100b.s3.yandex.net/vocab/voc_100b.sp

cd ../weights
curl --remote-name-all https://yalm-100b.s3.yandex.net/weights/layer_{00,01,[03-82],84}-model_00-model_states.pt
