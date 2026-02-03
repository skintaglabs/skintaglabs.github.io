# SkinTag Public Frontend

This directory contains both frontend versions for GitHub Pages deployment.

## Directory Structure

```
public/
├── index.html          # Landing page with version selector
├── classic/            # Simple HTML/CSS/JS version
│   ├── index.html      # Self-contained single-page app
│   └── config.js       # API configuration (injected during deployment)
└── app/                # Modern Next.js version (copied from build)
    └── (Next.js static export)
```

## Deployment Structure on GitHub Pages

When deployed, the site structure is:

```
https://[org].github.io/SkinTag/
├── /                   # Landing page (index.html)
├── /classic/           # Classic version
└── /app/               # Modern Next.js version
```

## Versions

### Classic (`/classic/`)
- **Technology**: Plain HTML, CSS, JavaScript
- **Features**:
  - Lightweight, single-page application
  - Warm, accessible design aesthetic
  - No build step required
  - Works with or without API URL configured
  - ABCDE self-check criteria
- **Use Case**: Fast loading, minimal resources, broader compatibility

### Modern (`/app/`)
- **Technology**: Next.js, React, TypeScript, Tailwind CSS, Framer Motion
- **Features**:
  - Smooth animations and transitions
  - Mobile camera integration
  - Progressive loading states
  - Enhanced visual feedback
- **Use Case**: Rich user experience, modern browsers

## Local Development

### Classic Version
```bash
# Serve from public directory
python -m http.server 8080 --directory public

# Or use any static server
npx serve public
```

Access at: `http://localhost:8080/classic/`

### Modern Version
See `app/skintag-webapp/README.md` for Next.js development

## API Configuration

Both versions can be configured to use external inference APIs.

### Classic
API URL can be set via:
1. **Deployment**: `config.js` is updated during GitHub Actions workflow
2. **Runtime**: URL parameter `?api=https://api.example.com`
3. **Default**: `http://localhost:8000` for local development

### Modern
API URL is set via environment variable during build:
```bash
NEXT_PUBLIC_API_URL=https://api.example.com npm run build
```

## Deployment

The GitHub Actions workflow (`.github/workflows/deploy-webapp.yml`) automatically:
1. Builds the Next.js app
2. Copies both versions to `dist/`
3. Injects API_URL into classic `config.js`
4. Deploys to GitHub Pages

Manual deployment is not recommended. Push to `main` branch to trigger automatic deployment.
