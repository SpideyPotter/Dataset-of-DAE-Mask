# MSWDD2022 YOLO Conversion & Roboflow Upload

## Convert LabelMe → YOLO

```bash
pip install -r requirements.txt
python scripts/convert_to_yolo.py
```

Output goes to `yolo_dataset/` with an 80/10/10 train/valid/test split and 4 classes:

- `yellow_dwarf`
- `powdery_mildew`
- `scab`
- `stripe_rust`

## Upload to Roboflow

Set your API key (from https://app.roboflow.com/settings/api):

```bash
export ROBOFLOW_API_KEY="your_key_here"
python scripts/upload_roboflow.py
```

Optional flags:

```bash
python scripts/upload_roboflow.py \
  --project mswdd2022-wheat-diseases \
  --workspace your-workspace-slug \
  --zip
```

The `--zip` flag uses Roboflow's zip upload flow (faster for large datasets).

## Manual upload alternative

If you prefer the web UI, zip the converted dataset and upload at https://app.roboflow.com:

```bash
python scripts/convert_to_yolo.py
zip -r yolo_dataset.zip yolo_dataset
```
