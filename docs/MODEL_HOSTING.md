# Model Hosting

Fine-tuned model hosted at [DTanzillo/MedGemma540](https://huggingface.co/DTanzillo/MedGemma540).

## Download Models

```bash
export USE_HF_MODELS=true
export HF_TOKEN=your_token  # For private repos
make app
```

Auto-downloads fine-tuned SigLIP model from HF and caches in `~/.cache/huggingface/skintag/`.

## Upload Models

After training:

```bash
pip install huggingface_hub[cli]
huggingface-cli login
cd results/cache/finetuned_model
huggingface-cli upload YourOrg/YourModel . --repo-type model
```

## Config

- `USE_HF_MODELS=true` - Enable HF downloads
- `HF_REPO_ID` - Override repo (default: `DTanzillo/MedGemma540`)
- `HF_TOKEN` - For private repos
