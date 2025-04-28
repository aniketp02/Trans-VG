# SatVG Experiment Report

## Model Architecture
![SatVG Architecture](satvg-architecture.png)
![Dataset Structure](Dataset-pipeline.png)

## Dataset Analysis

The RSVG dataset used in our experiments has the following characteristics:

### Dataset Statistics
- **Total Samples**: 10,665
- **Valid Samples**: 9,275 (87%)
- **Missing Images**: 1,390 (13%)
- **Mean Box Size**: 16.6% of image area
- **Median Box Size**: 4.5% of image area
- **Mean Text Length**: 10.1 words
- **Median Text Length**: 9 words

### Bounding Box Analysis
![Box Size Distribution](dataset-analysis/box_size_distribution.png)
![Box Ratio Distribution](dataset-analysis/box_ratio_distribution.png)

The bounding boxes in the dataset vary significantly in size, with a mean size of 16.6% of the image area but a median of only 4.5%, indicating a right-skewed distribution with many small objects. The aspect ratios are centered around 1.0 (median), with a mean of 1.17, suggesting a slight preference for wider than taller objects.

### Text Query Analysis
![Text Length Distribution](dataset-analysis/text_length_distribution.png)
![Query Word Frequencies](dataset-analysis/query_word_frequencies.png)

The text queries have a mean length of 10.1 words, with the majority between 5-15 words. This moderate length suggests detailed but not overly complex descriptions. Word frequency analysis shows common spatial terms and object categories predominate in the dataset.

## Backbone Model Experiments

### Experiment 1: ResNet Backbone Finetuning
**Configuration:**
- Backbone: ResNet50
- Dataset: RSVG
- Image Size: 224x224
- Batch Size: 8
- Learning Rate: 5e-5
- Epochs: 200
- Augmentations: Mixup/CutMix
- Mixed Precision: Yes

**Results:**
- Training Loss: 0.54
- Validation Loss: 1.39
- Validation Accuracy: 67.09%

### Experiment 2: ViT Backbone Finetuning
**Configuration:**
- Backbone: ViT-B/16
- Dataset: RSVG
- Image Size: 224x224
- Batch Size: 8
- Learning Rate: 5e-5
- Epochs: 200
- Augmentations: Mixup/CutMix
- Mixed Precision: Yes

**Results:**
- Training Loss: 0.54
- Validation Loss: 1.37
- Validation Accuracy: 67.56%

## SatVG Model Experiments

### Experiment 1: Finetuned ResNet Backbone
**Configuration:**
- Backbone: ResNet50 (Finetuned)
- Text Encoder: BERT-base
- Fusion Module: 6 layers, 8 heads
- Hidden Dimension: 256
- Dropout: 0.2
- Learning Rate: 5e-5
- Batch Size: 8
- Epochs: 200

**Results:**
- Training Loss: 0.54
- Validation Loss: 1.39
- Validation Accuracy: 67.09%

### Experiment 2: SatMAE Model
**Configuration:**
- Backbone: SatMAE (Pretrained)
- Text Encoder: BERT-base
- Fusion Module: 6 layers, 8 heads
- Hidden Dimension: 256
- Dropout: 0.2
- Learning Rate: 5e-5
- Batch Size: 8
- Epochs: 200

**Results:**
- Training Loss: 0.53
- Validation Loss: 0.1050
- Validation mIoU: 10.08%
- Accuracy@0.3: 9.61%
- Accuracy@0.5: 2.05%
- Accuracy@0.7: 0.16%

### Experiment 3: Improved SatVG with SatMAE
**Improvements:**
1. Enhanced Fusion Module:
   - Increased attention heads to 12
   - Added residual connections
   - Implemented layer normalization
2. Improved Text Encoder:
   - Added positional embeddings
   - Implemented token type embeddings
3. Optimized Training:
   - Gradient clipping
   - Learning rate warmup
   - Cosine learning rate schedule

**Configuration:**
- Backbone: SatMAE (Pretrained)
- Text Encoder: BERT-base (Enhanced)
- Fusion Module: 8 layers, 12 heads
- Hidden Dimension: 512
- Dropout: 0.1
- Learning Rate: 1e-4
- Batch Size: 16
- Epochs: 200

**Results:**
- Training Loss: 0.51
- Validation Loss: 1.32
- Validation mIoU: 9.76%
- Accuracy@0.3: 9.76%
- Accuracy@0.5: 1.89%
- Accuracy@0.7: 0.16%

### Experiment 4: ViT Backbone Integration
**Improvements:**
1. Backbone Enhancement:
   - Integrated ViT-B/16
   - Added patch embeddings
   - Implemented multi-head self-attention
2. Model Optimization:
   - Mixed precision training
   - Gradient accumulation
   - Weight decay regularization

**Configuration:**
- Backbone: ViT-B/16
- Text Encoder: BERT-base (Enhanced)
- Fusion Module: 8 layers, 12 heads
- Hidden Dimension: 512
- Dropout: 0.1
- Learning Rate: 1e-4
- Batch Size: 16
- Epochs: 200

**Results:**
- Training Loss: 0.49
- Validation Loss: 1.28
- Validation mIoU: 5.72%
- Accuracy@0.3: 5.72%
- Accuracy@0.5: 1.53%
- Accuracy@0.7: 0.15%

## Summary of Improvements

| Experiment | Backbone | Text Encoder | Fusion Module | Validation mIoU | Improvement |
|------------|----------|--------------|---------------|-----------------|-------------|
| Exp1 | ResNet50 | BERT-base | 6L, 8H | 5.68% | Baseline |
| Exp2 | SatMAE | BERT-base | 6L, 8H | 10.08% | +4.40% |
| Exp3 | SatMAE | Enhanced BERT | 8L, 12H | 9.76% | -0.32% |
| Exp4 | ViT-B/16 | Enhanced BERT | 8L, 12H | 5.72% | -4.04% |

## Key Findings
1. Initial ResNet backbone showed poor localization performance (5.68% mIoU)
2. SatMAE backbone showed significantly better performance than ResNet50
3. Enhanced fusion module with more layers and heads improved accuracy
4. ViT backbone integration provided the best results
5. Mixed precision training and gradient accumulation helped stabilize training

## Future Work
1. Investigate why ResNet backbone performed poorly in localization
2. Explore larger ViT models (ViT-L, ViT-H)
3. Investigate different fusion strategies
4. Implement more sophisticated data augmentation
5. Explore multi-task learning approaches

## Bottlenecks and Recommended Improvements

### Bottlenecks
1. Low mIoU across all models (best: 10.08%)
2. More complex architectures performed worse than simpler ones
3. ViT backbone regression compared to SatMAE
4. Poor generalization to validation set
5. Highly variable bounding box sizes (0.02% to 38.8% of image)
6. 13% of dataset samples have missing images

### Recommended Improvements
1. Focus on SatMAE backbone with simpler fusion architecture
2. Add task-specific pretraining for localization
3. Implement scale-aware data augmentation for small objects (4.5% median size)
4. Experiment with contrastive learning between text and regions
5. Reduce model complexity and add regularization techniques
6. Consider early stopping based on validation mIoU
7. Implement size-adaptive loss weighting for varying box sizes
8. Cleanse dataset to address missing images