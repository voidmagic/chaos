A minimum running example of parameter differentiation.

Requirement:
```
apex
fairseq
scikit-learn
pytorch
```



1. Process data following `https://github.com/pytorch/fairseq`.
2. Training:
```
data_bin=    # data path 
lang_pairs=  # comma separated language pairs

fairseq-train $data_path \
    --task parameter_differentiation_task --lang-pairs $lang_pairs --encoder-langtok tgt \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --lr 0.0015 --adam-betas '(0.9,0.98)' \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --arch parameter_differentiation_base_model \
    --max-tokens 8192 \
    --user-dir $PWD 
```

3. Decoding
```
source_lang=
target_lang=

fairseq-generate $data_path \
    --task parameter_differentiation_task --lang-pairs $lang_pairs --encoder-langtok tgt \
    --beam 4 --lenpen 0.6 --remove-bpe sentencepiece \
    --source-lang $source_lang --target-lang $target_lang > result.$source_lang-$target_lang.txt
```



