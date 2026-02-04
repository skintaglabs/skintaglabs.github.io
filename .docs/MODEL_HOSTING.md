# Model Hosting

Fine-tuned model hosted at [skintaglabs/siglip-skin-lesion-classifier](https://huggingface.co/skintaglabs/siglip-skin-lesion-classifier).

## Download Models

```bash
export USE_HF_MODELS=true
make app
```

Auto-downloads fine-tuned SigLIP model from HF and caches in `~/.cache/huggingface/skintag/`.

**Note:** HF_TOKEN only needed for private repos.

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
- `HF_REPO_ID` - Override repo (default: `skintaglabs/siglip-skin-lesion-classifier`)
- `HF_TOKEN` - For private repos
