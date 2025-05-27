# Fine tuning model

A simple package for fine tuning pretrained model

## Default settings:

- Qwen 1.5 - 0.5B parameters
- Fine tune using LoRA
- Dataset: OpenAssistant/oasst1 and WildChat (English only)

Tested on 3080Ti

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To change model, edit settings.py

To change dataset, edit dataloader.py

To start training:

```bash
python model.py
```

Inference:

```bash
python inference.py
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
