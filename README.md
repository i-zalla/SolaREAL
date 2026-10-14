# SolaREAL Streamlit Platform

This project wraps the client's provided deepfake inference pipeline in a Streamlit UI.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

FFmpeg must also be installed and available on PATH.

## Model files

Put these files in `models/`:

- `attention_fusion.pth`
- `best_model.pth`
- `scaler.save`
- `yolov8m-face-lindevs.pt`

They are already copied into this project when they were available in the working directory.

## Important

The implementation in `src/pipeline.py` follows the supplied `test_single_video.py` flow:

video -> audio/visual feature extraction -> AttentionFusion -> scaler -> AV-MSTCN -> fake/real probabilities.

The original client script is retained as `client_reference_test_single_video.py` for comparison.

## Hugging Face models

The first run loads:

- `google/vit-base-patch16-224-in21k`
- `facebook/wav2vec2-xls-r-300m`

These may be downloaded/cached by Transformers on first use.
