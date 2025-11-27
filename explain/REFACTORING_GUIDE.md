# Quick Reference: Explain Folder Refactoring

## Mathematical Formula

### Gradient of Similarity Measure w.r.t. Input Features

```
∂s_j/∂X = (V_j Λ Q_j^T) (S^{-1} R_j^T)
```

**Components:**
- `V_j = w_j Q_{w,j}` - Rotated prototype
- `Λ = diag(λ_1, ..., λ_d)` - Relevance matrix  
- `Q_j^T` - Transpose of rotation matrix from SVD
- `S^{-1}` - Inverse singular values
- `R_j^T` - Right singular vectors

**In code:**
```python
# Left term: (nprotos, D, n)
V = torch.bmm(prototypes, Qw)
H_left = torch.bmm(torch.matmul(V, Lamda), Qt)

# Right term: (1, d, n)
S_inv = torch.diag_embed(1 / s_matrix)
H_right = torch.bmm(S_inv, Rt_matrix)

# Gradient: (nprotos, D, n)
sim_grad = torch.matmul(H_left, H_right[0])

# Reshape to spatial: (nprotos, C, H, W)
sim_grad = sim_grad.view(nprotos, num_channels, width, height)
```

---

## Before vs After

### Before (Batch Processing):
```python
# Load ALL images at once
images_names, labels, transformed_images = load_and_process_images(args, logger)

# Process in batch
for idx, (img_name, label, sample) in enumerate(zip(images_names, labels, transformed_images)):
    # Process each image...
```

**Issues:**
- ❌ Loads all images into memory
- ❌ Loop overwrites results (only last image processed)
- ❌ Incorrect `torch.exp()` transformation
- ❌ Dimension mismatches

### After (One-by-One Processing):
```python
# Create generator (lazy loading)
image_generator = load_and_process_images_generator(args, logger)

# Process one-by-one
for img_name, label, img_transformed, original_image in image_generator:
    compute_single_image_heatmap(model, img_name, img_transformed, label, ...)
```

**Benefits:**
- ✅ Memory efficient (constant memory usage)
- ✅ Correct gradient computation
- ✅ Clean, modular code
- ✅ Better logging and error handling

---

## Key Functions

### 1. Compute Gradient (importance_scores.py)

```python
sim_grad = compute_feature_importance(
    feature,          # (1, C, H, W) - Feature map from CNN
    Rt,               # (1, d, n) - Right singular vectors
    S,                # (1, d) - Singular values
    output,           # Dict with 'Q', 'Qw' rotation matrices
    prototypes,       # (nprotos, D, d) - Prototype subspaces
    relevances,       # (1, d) - Relevance weights
    target_class=0    # Optional: compute for specific class
)
# Returns: (C, H, W) if target_class specified, else (nprotos, C, H, W)
```

### 2. Process Single Image (feature_importance.py)

```python
compute_single_image_heatmap(
    model=model,
    img_name="image.jpg",
    img_transformed=img_tensor,  # (C, H, W)
    label=torch.tensor(0),       # Scalar
    logger=logger,
    args=args,
    original_image=pil_image     # Optional for overlay
)
# Saves: heatmap.png, heatmap_raw.npy, overlay.png
```

### 3. Load Images (data_utils.py)

```python
# Generator - yields one image at a time
for filename, label, transformed_img, original_img in load_and_process_images_generator(args, logger):
    # Process image...
```

---

## Output Files

```
explanations/
└── BRAIN/                          # Dataset name
    ├── image_001/
    │   ├── image_001.jpg          # Original image
    │   ├── heatmap.png            # Importance heatmap (colorized)
    │   ├── heatmap_raw.npy        # Raw importance values
    │   └── overlay.png            # Heatmap + original image
    └── image_002/
        └── ...
```

---

## Usage Example

```bash
# Run explanation
python main_explain_decision.py \
    --model_path ./run_prototypes/ \
    --sample_dir ./samples \
    --results_dir ./explanations
```

**Directory structure expected:**
```
samples/
└── BRAIN/                  # args.dataset
    ├── img_gl_001.jpg     # Images to explain
    ├── img_me_002.jpg
    └── ...

run_prototypes/
├── metadata/
│   └── args.pickle        # Model arguments
└── checkpoints/
    └── best_test_model    # Model weights
```

---

## Debugging Tips

### Check gradient shapes:
```python
print(f"Feature map: {feature.shape}")        # (1, C, H, W)
print(f"Rt matrix: {Rt.shape}")               # (1, d, n)
print(f"S matrix: {S.shape}")                 # (1, d)
print(f"Gradient: {sim_grad.shape}")          # (nprotos, C, H, W) or (C, H, W)
print(f"Importance map: {importance_map.shape}")  # (H, W)
```

### Verify no NaN/Inf:
```python
assert not torch.isnan(sim_grad).any(), "NaN in gradient!"
assert not torch.isinf(sim_grad).any(), "Inf in gradient!"
```

### Check prototype count:
```python
nprotos = model.prototype_layer.xprotos.shape[0]
nclasses = args.nclasses
assert nprotos == nclasses, f"Expected {nclasses} prototypes, got {nprotos}"
```

---

## Common Issues & Solutions

### Issue: "Expected batch_size=1, got X"
**Solution:** Ensure you're passing single image with `.unsqueeze(0)`

### Issue: Dimension mismatch in gradient computation
**Solution:** Check that all matrices have correct shapes (see above)

### Issue: Memory error
**Solution:** Use the generator pattern (already implemented)

### Issue: Heatmap looks wrong
**Solution:** 
1. Check gradient computation (no `exp()`)
2. Verify normalization in `save_feature_importance_heatmap()`
3. Ensure upsampling to correct size

---

## Performance

**Memory Usage:**
- Before: O(N × C × H × W) - all images in memory
- After: O(C × H × W) - constant per image

**Processing:**
- Sequential: One image at a time
- Can be parallelized if needed (modify generator)

---

## Next Steps

1. ✅ Test on sample images
2. ✅ Verify heatmaps are correct
3. ✅ Compare with previous implementation
4. Optional: Add support for batch processing (if needed)
5. Optional: Add more visualization options (different colormaps, etc.)
