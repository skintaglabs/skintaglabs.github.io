# Deployment Guide

Quick guide to deploying SkinTag.

## Prerequisites

1. **Hugging Face token** (required - repo is gated)
   - Get token: https://huggingface.co/settings/tokens
   - Accept repo terms: https://huggingface.co/skintaglabs/siglip-skin-lesion-classifier

2. **Add GitHub secrets**:
   ```bash
   gh secret set HF_TOKEN
   # Paste your token when prompted
   ```

## Deploy Inference Server

Run the "Inference Server" workflow:
```bash
gh workflow run inference-server.yml
```

This will:
- Start inference server in GitHub Actions (free tier)
- Load model from Hugging Face
- Create Cloudflare tunnel (no setup needed)
- Auto-deploy frontend with tunnel URL
- Run for 5 hours, then auto-restart

## Deploy Frontend Only

The frontend auto-deploys when:
- You push changes to `webapp/`
- Inference server starts (with tunnel URL)

Manual deployment:
```bash
gh workflow run deploy-webapp.yml
```

View at: https://medgemma540.github.io/SkinTag/

## Local Development

```bash
# Start backend
make app

# Serve frontend
cd webapp
python -m http.server 8080

# Visit: http://localhost:8080?api=http://localhost:8000
```

## Technical Details

See `.docs/` for:
- Model hosting internals
- Serverless inference architecture
- Research paper outline
