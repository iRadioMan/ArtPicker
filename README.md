# ArtPicker

ArtPicker downloads images from Derpibooru and sorts them by how closely their visual features match the user's preferences. It uses CLIP to extract image features and a binary classifier trained on liked and disliked examples.

## Directories

- `input/liked/` — images the user likes.
- `input/disliked/` — images the user dislikes.
- `temp/` — downloaded images waiting for analysis.
- `output/good/` — images classified as likely to be liked.
- `output/bad/` — images classified as unlikely to be liked.
- `model/` — saved classifier checkpoints.

The classifier learns from visual features. It does not store the training images' file names. It needs at least one readable image in each input class; more balanced and varied examples generally give it better feedback.

## Setup

From the project root, create a virtual environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first training or analysis run may download the configured CLIP model. Internet access is required for this and for image downloads.

## Workflow

1. Put examples into `input/liked/` and `input/disliked/`.
2. Train or refresh the preference classifier:

   ```powershell
   python .\train.py
   ```

3. Download candidates by search tags. The configured `api.max_limit` is the default; `--limit` overrides it:

   ```powershell
   python .\download.py "safe, sunset" --limit 20
   ```

4. Classify every supported image currently in `temp/`:

   ```powershell
   python .\analyze.py
   ```

   Images with a predicted like probability at or above `analysis.match_threshold` go to `output/good/`; the rest go to `output/bad/`. Analysis moves files out of `temp/`. If an image used for training is still in `temp/`, it is also treated as a candidate and classified.

All scripts accept `--config PATH` to use a different JSON configuration file.

## Configuration

Edit `config.json` to change the CLIP model, classifier regularization, match threshold, or folder paths. `analysis.sensitivity` (0–4) controls how strongly training emphasizes visual features that distinguish liked examples from disliked ones. Retrain after changing sensitivity.

`images.max_pixels` sets Pillow's decompression-bomb pixel limit. Its default of 200 million pixels accepts high-resolution images such as 96-megapixel art while retaining a limit for exceptionally large files. Increasing it also increases the memory required to decode images.
