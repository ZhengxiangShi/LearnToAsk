# Learning to Execute Actions or Ask Clarification Questions
Here are the dataset and codes for the Findings of NAACL paper titled "[**Learning to Execute Actions or Ask Clarification Questions**](https://arxiv.org/abs/2204.08373)". 

## Dependencies

```
python==3.7
torch==1.7
tensorboardX
prettytable
```

## Introduction
An intelligent agent should not only understand and execute the instructor's requests but also be able to take initiatives, e.g., asking clarification questions, in case the instructions are ambiguous.

<p align="center">
    <img src="Asset/example.png" width="550">
</p>
<p align="center">
    <b>A simple example of builder task. </b>
</p>

## Code and Dataset
- `CollaborativeBuilding`: Codes for collaborative building task;
- `LearnToAsk`: Codes for learning to ask task and joint learning task;
- `builder_utterance_labels.json`: Annotations of all builder utterances. Please ignore `builder_utterance_labels.txt`, which is our draft version.
- The raw dataset `Minecraft Dialogue Corpus` is from the [repository](https://github.com/prashant-jayan21/minecraft-bap-models#raw-data).

### 1. Dataset Preparation
Please download the [original dataset](https://drive.google.com/file/d/1jAu_LymRSlqNJznViJZoi7xf8O9V6pUo/view?usp=sharing). Then
```
unzip data.zip
cd data
wget https://nlp.stanford.edu/data/glove.42B.300d.zip
unzip glove.42B.300d.zip
cd ../CollaborativeBuilding/builder
python vocab.py --lower --use_speaker_tokens --oov_as_unk --all_splits --add_builder_utterances
cd ..
```

### 2. Code
Please run codes in `CollaborativeBuilding` and `LearnToAsk` for Collaborative Building task and Learning to Ask task respectively.

## Citation
Please cite our work if it is helpful.
```
@inproceedings{Shi2022learning,
title = {Learning to Execute Actions or Ask Clarification Questions},
author = {Shi, Zhengxiang and Feng, Yue and Lipani, Aldo},
year = {2022},
address = {Seattle, Washington, USA},
booktitle = {Findings of the North American Chapter of the Association for Computational Linguistics},
publisher = {Association for Computational Linguistics},
keywords = {Conversational System, Clarification Questions},
url = {https://arxiv.org/abs/2204.08373}
}
