# BERT-based Chinese Word Segmentation

This project implements a Chinese word segmentation model based on BERT with a CRF layer. It builds upon the existing BiLSTM-CRF model to offer improved performance through contextual embeddings from BERT.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Training

To train the BERT-based word segmentation model:

```bash
python run_bert.py --cuda  # Add --cuda if you have a GPU
```

Training parameters can be customized:

- `--lr`: Learning rate (default: 2e-5)
- `--max_epoch`: Maximum number of epochs (default: 5)
- `--batch_size`: Batch size for training (default: 16)
- `--max_length`: Maximum sequence length (default: 512)
- `--dropout`: Dropout rate (default: 0.1)
- `--weight_decay`: Weight decay for optimizer (default: 0.01)
- `--early_stopping`: Number of epochs for early stopping (default: 2)
- `--warmup_steps`: Warmup steps for learning rate scheduler (default: 500)

Example:

```bash
python run_bert.py --cuda --lr 3e-5 --batch_size 8 --max_epoch 10
```

## Inference

To use the trained model for word segmentation:

```bash
python infer_bert.py --model_path save/best_bert_model.pkl
```

This will process the text in `data/test_final.txt` and save the segmented results to `cws_result_bert.txt`.

## Model Architecture

The model consists of:

1. **BERT Encoder**: Uses the pre-trained `bert-base-chinese` model to obtain contextual embeddings
2. **Linear Layer**: Maps BERT outputs to tag space
3. **CRF Layer**: Performs sequence labeling using Conditional Random Fields

## Comparison with BiLSTM-CRF

The BERT-CRF model offers several advantages over the BiLSTM-CRF approach:

1. **Contextual Understanding**: BERT captures contextual information more effectively
2. **Pre-trained Knowledge**: Leverages knowledge from pre-training on large Chinese corpora
3. **Character-level Information**: Better handles rare characters through subword embeddings

## Training Data

The model is trained on the same data as the BiLSTM-CRF model, using the BMES tagging scheme:
- B: Beginning of a word
- M: Middle of a word
- E: End of a word
- S: Single character word

## Acknowledgements

- This project uses the `bert-base-chinese` model from Hugging Face Transformers
- The CRF implementation uses the `pytorch-crf` package 