# Constrained APE

A fork from fairseq

## Things added
- **fairseq/data/ape_dataset.py**: based on LanguagePairDataset. Added support for mt, term, and others
- **fairseq/data/ape_lev.py**: based on trans_lev. Added support for custom initialization for *prev_target* for training and *init_tokens* at test time
  - Parameters:
    - **--input-type**: input to be fed into the encoder. Choices are *src-only*, *concatenate*, and *multisource*. *Concatenate* combines src and mt with <sep>, and *multisource* will change the input into a dictionary of src and mt (required by MultisourceEncoder).
    - **--prev-target**: initialization for *prev_target* for imitation learning. Choices are *target*, *mt*, and *term*.
    - **--init-output**: Initialization of starting string at test time. Choices are *blank*, *mt*, and *term*.
- **fairseq/models/multisource_encoder.py**: Encoder similiar to CompositeEncoder, but takes in separate input for each encoder and concatenate the outputs of all encoders before passing into the decoder.
- **fairseq/models/nat/multisource_levenshtein_transformer.py**: LevenshteinTransformer with MultisourceEncoder
  - Parameters:
    - **--share-encoder**: Using the same encoder for src_encoder and mt_encoder
- **fairseq/models/multisource_transformer.py**: Transformer with MultisourceEncoder
  - Parameters:
    - **--share-encoder**: Using the same encoder for src_encoder and mt_encoder

## Examples to run the Models

### Levenshtein Transformer training from noisy_mt -> target
```
fairseq-train \
    $TRAIN_DIR \
    --save-dir checkpoints/levenshtein_transformer_mt_prev_target \
    --ddp-backend=no_c10d \
    --task ape_lev \
    --criterion nat_loss \
    --arch levenshtein_transformer \
    --noise random_delete \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --fixed-validation-seed 7 \
    --max-tokens 8000 --update-freq 8 \
    --save-interval-updates 10000 --no-epoch-checkpoints \
    --max-update 300000 --source-lang src --target-lang pe --fp16 \
    --num-workers 16 --prev-target mt\
```

### Multisource Levenshtein Transformer
```
fairseq-train \
    $TRAIN_DIR \
    --save-dir checkpoints/multisource_levenshtein_transformer \
    --ddp-backend=no_c10d \
    --task ape_lev \
    --criterion nat_loss \
    --arch multisource_levenshtein_transformer \
    --noise random_delete \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --fixed-validation-seed 7 \
    --max-tokens 8000 --update-freq 8 \
    --save-interval-updates 10000 --no-epoch-checkpoints \
    --max-update 300000 --source-lang src --target-lang pe --fp16 \
    --num-workers 16 --input-type multisource

```

### Multisource Transformer
```
fairseq-train \
    $TRAIN_DIR \
    --save-dir checkpoints/multisource_transformer \
    --arch multisource_transformer --share-decoder-input-output-embed \
    --task ape \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 --update-freq 8 \
    --save-interval-updates 10000 --no-epoch-checkpoints \
    --max-update 300000 --source-lang src --target-lang pe --fp16 \
    --num-workers 16 --input-type multisource
```
