#!/bin/pwsh

New-Item -Type Directory -Path yalm100b_checkpoint/vocab -Force
New-Item -Type Directory -Path yalm100b_checkpoint/weights -Force

cd yalm100b_checkpoint/vocab

Invoke-WebRequest -Uri https://yalm-100b.s3.yandex.net/vocab/voc_100b.sp -OutFile voc_100b.sp

cd ../weights

$skip_layers = 2, 83
for ($layer = 0; $layer -le 84; $layer++) {
    if ($skip_layers -contains $layer) {
        continue
    }
    $layer_str = $layer.ToString('00')
    $layer_file_name = "layer_${layer_str}-model_00-model_states.pt"
    $layer_url = "https://yalm-100b.s3.yandex.net/weights/${layer_file_name}"
    Invoke-WebRequest -Uri $layer_url -OutFile $layer_file_name
}
