# Probabilistic LVQ on Grassmann Manifold

This repository contains the implementation of Probabilistic Learning Vector Quantization (LVQ) on the Grassmann manifold.

## Training

To train the model, use `main.py`:

```bash
python main.py --dataset PETS --nclasses 37 --log_dir ./run_prototypes
```

## Explanation

To generate explanations for model decisions, use `main_explain.py`.

### Region-based Explanations

Generate heatmaps and visualizations for specific images:

```bash
python main_explain.py --mode regions --dataset PETS --model_path ./run_prototypes/pca/cub/run2/
```

### Patch Finding (Optional)

Find the closest patches in the training set to each principal direction of the prototypes:

```bash
python main_explain.py --mode regions --find_patches
```

> [!TIP]
> If you omit `--dataset` or `--model_path`, the script will automatically use the values defined in `explain/config/regions_config.yaml`. You can also override these values via CLI:
> ```bash
> python main_explain.py --mode regions --find_patches --dataset PETS --model_path ./path/to/model
> ```

- **`--find_patches`**: This flag is **optional**. 
    - If **set**, the script will iterate through the training set to find and save the most similar patches for each class and prototype direction.
    - If **not set**, the script will perform the standard region-based explanation for the images in the sample directory.
- **`--num_images`**: (Optional) Limit the number of images processed during patch finding or explanation.

The results will be saved in `result_dir/dataset_name/patchs/` when using `--find_patches`.
