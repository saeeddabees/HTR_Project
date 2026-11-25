# Learning-rate study (attention model, 10 epochs, batch 8)

| Run                      |     LR |   Seed |   Best epoch |   Best val CER % |   Val WER % at best |   Train time s |
|:-------------------------|-------:|-------:|-------------:|-----------------:|--------------------:|---------------:|
| hp_attention_lr0005      | 0.0005 |      0 |            9 |            16.68 |               50.1  |          684.2 |
| hp_attention_lr001       | 0.001  |      0 |           10 |            16.2  |               50.84 |          670.7 |
| hp_attention_lr002       | 0.002  |      0 |            9 |            15    |               47.54 |          679.5 |
| hp_attention_lr002_seed1 | 0.002  |      1 |            9 |            15.06 |               46.96 |          671.5 |
| hp_attention_lr002_seed2 | 0.002  |      2 |           10 |            14.17 |               45.44 |          685.5 |
