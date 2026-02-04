# Retraining Recommendations for Clinical Triage Gradation

## Current Performance Gaps

Based on the clinical triage analysis, the existing models have these limitations:

| Task | Current | Target | Gap |
|------|---------|--------|-----|
| Binary malignancy (95% sens) | 77% specificity | 85%+ specificity | -8% |
| Melanoma detection | 68% sensitivity | 95% sensitivity | -27% |
| SCC detection | 59% sensitivity | 93% sensitivity | -34% |
| Non-neoplastic identification | 59% sensitivity | 75% sensitivity | -16% |

**Root cause**: The SigLIP embeddings were not fine-tuned for condition-level discrimination. The binary classifier works well (AUC 0.948) because it only needs to separate two broad groups.

---

## Recommended Retraining Strategy

### Option 1: Hierarchical Multi-Task Fine-Tuning (Recommended)

Fine-tune SigLIP with a hierarchical classification head:

```
SigLIP Backbone
    │
    ├── Binary Head: Malignant vs Benign
    │
    └── Condition Head: 10-class with class weights
            │
            ├── URGENT conditions (melanoma, SCC): weight 3.0
            ├── PRIORITY conditions (BCC, AK): weight 2.0
            ├── ROUTINE conditions (non-neoplastic): weight 1.5
            └── MONITOR conditions (benign): weight 1.0
```

**Training recipe**:
```python
# Multi-task loss with clinical weighting
loss = (
    0.5 * binary_loss +                    # Primary triage signal
    0.3 * condition_loss_weighted +        # Condition estimation
    0.2 * contrastive_loss                 # Separate embedding clusters
)

# Class weights inversely proportional to sample count
# but boosted for high-priority conditions
condition_weights = {
    'melanoma': 15.0,      # rare + critical
    'scc': 12.0,           # rare + urgent
    'bcc': 3.0,            # common but priority
    'ak': 8.0,             # moderate rarity
    'non_neoplastic': 1.5, # common, needs separation
    'nevus': 0.5,          # very common
    # ...
}
```

**Expected improvement**: +15-25% condition sensitivity, +5-10% specificity at 95% sensitivity.

---

### Option 2: Focal Loss for Hard Examples

The current cross-entropy loss treats all errors equally. For clinical triage, missing a melanoma is catastrophic while over-flagging a benign lesion is acceptable.

**Implementation**:
```python
class ClinicalFocalLoss(nn.Module):
    """Focal loss with clinical asymmetry."""

    def __init__(self, gamma=2.0, false_negative_penalty=5.0):
        self.gamma = gamma
        self.fn_penalty = false_negative_penalty  # for malignant classes

    def forward(self, pred, target):
        # Standard focal loss
        pt = pred.gather(1, target.unsqueeze(1)).squeeze()
        focal_weight = (1 - pt) ** self.gamma

        # Extra penalty for false negatives on malignant
        is_malignant = target < 4  # conditions 0-3 are malignant
        penalty = torch.where(is_malignant, self.fn_penalty, 1.0)

        loss = -penalty * focal_weight * torch.log(pt + 1e-8)
        return loss.mean()
```

---

### Option 3: Embedding Space Optimization via Contrastive Learning

Train embeddings to cluster by clinical tier, not just condition:

```python
# SuperCon-style loss for clinical tiers
def clinical_contrastive_loss(embeddings, tiers):
    """Pull together samples from same tier, push apart different tiers."""
    # Tier 0: URGENT (melanoma, SCC)
    # Tier 1: PRIORITY (BCC, AK)
    # Tier 2: ROUTINE (non-neoplastic)
    # Tier 3: MONITOR (benign)

    # Critical: URGENT must be maximally separated from MONITOR
    urgent_mask = tiers == 0
    monitor_mask = tiers == 3

    # Minimize distance within tier, maximize distance between urgent/monitor
    intra_tier_loss = pairwise_distance(embeddings[urgent_mask]).mean()
    inter_tier_loss = -pairwise_distance(
        embeddings[urgent_mask], embeddings[monitor_mask]
    ).mean()

    return intra_tier_loss + 2.0 * inter_tier_loss
```

---

### Option 4: Calibration Fine-Tuning

Even without full retraining, calibrating the condition classifier can improve clinical utility:

```python
# Temperature scaling per condition
calibrated_probs = softmax(logits / temperature[condition_id])

# Learn per-condition temperatures to match clinical sensitivity targets
# Lower temperature = sharper predictions = higher sensitivity
target_temperatures = {
    'melanoma': 0.5,   # very sharp - flag any suspicion
    'scc': 0.6,
    'bcc': 0.8,
    'nevus': 1.5,      # soft - need high confidence to reassure
}
```

This can be done on held-out validation data without full model retraining.

---

## Practical Implementation Plan

### Phase 1: Quick Wins (No Retraining)

1. **Calibration tuning**: Learn per-condition temperature scaling (1-2 hours)
2. **Threshold optimization**: Use clinical analysis results to set optimal thresholds
3. **Ensemble**: Combine XGBoost binary + logistic condition for better coverage

### Phase 2: Classifier Retraining (1-2 days)

1. Retrain condition classifier with class weights from clinical analysis
2. Use focal loss instead of cross-entropy
3. Add tier-level auxiliary loss

### Phase 3: Full SigLIP Fine-Tuning (3-5 days)

1. Unfreeze last 4 transformer layers
2. Multi-task training: binary + condition + contrastive
3. Progressive unfreezing schedule
4. Evaluate on held-out clinical test set

---

## Expected Outcomes

| Metric | Current | After Phase 1 | After Phase 3 |
|--------|---------|---------------|---------------|
| Binary AUC | 0.948 | 0.950 | 0.965 |
| Melanoma sensitivity | 68% | 80% | 92% |
| SCC sensitivity | 59% | 75% | 88% |
| Specificity @ 95% sens | 77% | 80% | 87% |
| Condition macro-F1 | 0.60 | 0.68 | 0.78 |

---

## Data Augmentation Considerations

For the new gradation strategy, augmentation should preserve clinical features:

```python
# Safe augmentations for dermoscopy
safe_augs = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=90, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
])

# Avoid: Heavy color jittering (destroys melanin patterns)
# Avoid: Aggressive cropping (may remove diagnostic features)
# Avoid: Noise injection (mimics artifacts, not pathology)
```

---

## Summary

**Most impactful change**: Hierarchical multi-task fine-tuning with clinical class weights. This preserves the strong binary performance while boosting condition-level discrimination, especially for high-priority conditions (melanoma, SCC).

**Quickest improvement**: Per-condition calibration + threshold optimization using the clinical analysis results already generated.
