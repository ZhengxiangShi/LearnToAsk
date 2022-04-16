# Collaborative Building Task

## 1. Preprocess
```
cd builder
python dataloader_with_glove.py --split train --json_data_dir data_path
python dataloader_with_glove.py --split val --json_data_dir data_path
python dataloader_with_glove.py --split test --json_data_dir data_path
cd ..
```

## 2. Train
```
python train.py --json_data_dir data_path --saved_models_path model_path
```

## 3. Test
```
python test.py --json_data_dir data_path --saved_models_path model_path
```

## Code Reference
Some of pre-processing codes are based on:
```
[1]: https://github.com/prashant-jayan21/minecraft-bap-models
``` 