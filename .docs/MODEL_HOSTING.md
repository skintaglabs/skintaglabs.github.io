# Model Hosting

Fine-tuned model hosted at [skintaglabs/siglip-skin-lesion-classifier](https://huggingface.co/skintaglabs/siglip-skin-lesion-classifier).

## Download Models (local / Docker)

```bash
export USE_HF_MODELS=true
make app
```

Auto-downloads the fine-tuned SigLIP model from HF and caches to `~/.cache/huggingface/skintag/`.

**Note:** `HF_TOKEN` is only needed if the model repo is private.

## Upload Models

After training:

```bash
pip install huggingface_hub[cli]
huggingface-cli login
cd results/cache/finetuned_model
huggingface-cli upload skintaglabs/siglip-skin-lesion-classifier . --repo-type model
```

## Config

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_HF_MODELS` | `false` | Set to `true` to download from HF on startup |
| `HF_REPO_ID` | `skintaglabs/siglip-skin-lesion-classifier` | Override the model repo |
| `HF_TOKEN` | — | Required only for private repos |

---

## Inference Server Deployment: Hugging Face Spaces

The inference backend runs as a Docker-based HF Space. The `README.md` contains the required Space YAML frontmatter (`sdk: docker`, `app_port: 8000`).

### First-time setup

1. **Create a Space** under the `skintaglabs` org at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: Docker
   - Visibility: Public

2. **Push this repo** to the Space (or link it via the Space's Git remote):
   ```bash
   git remote add space https://huggingface.co/spaces/skintaglabs/skintag-inference
   git push space main
   ```

3. **Set secrets** in Space Settings → Variables and secrets:
   - `USE_HF_MODELS` = `true`
   - `HF_TOKEN` = your HF token (if model repo is private)
   - `HF_REPO_ID` = `skintaglabs/siglip-skin-lesion-classifier`

HF Spaces builds the `Dockerfile` and starts the server on port 8000. The Space URL will be `https://skintaglabs-skintag-inference.hf.space`.

### Connecting the frontend

Add `API_URL` to the GitHub repo's Actions secrets (Settings → Secrets → Actions):
```
API_URL = https://skintaglabs-skintag-inference.hf.space
```

The `deploy-webapp.yml` workflow reads this secret and bakes it into the React build as `VITE_API_URL`.

### Updating the model

After uploading a new model version to HF, restart the Space (Settings → Factory reboot) to pull the updated weights on next startup.
