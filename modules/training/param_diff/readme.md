A minimum running example of parameter differentiation.

Requirement: fairseq, pytorch


1. Process data following `https://github.com/pytorch/fairseq`.
2. train a model:
```
fairseq-train \
    $data_path \
    --task parameter_differentiation_task \
    --criterion cross_entropy --optimizer adam \
    --arch parameter_differentiation_base_model \
    --max-tokens 4096 --lang-pairs $lang_pairs 
    --user-dir $PWD --fp16 --lr 0.0007
```
