# Deployment Guide

This guide explains how to deploy the Next.js frontend to Vercel using GitHub Actions.

## Prerequisites

1. A Vercel account
2. A GitHub repository with the code
3. The repository must have the necessary secrets configured

## Setup Steps

### 1. Create a Vercel Project

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your GitHub repository
4. Set the **Root Directory** to `frontend`
5. Configure the project settings:
   - Framework Preset: Next.js
   - Build Command: `pnpm build`
   - Output Directory: `.next`
   - Install Command: `pnpm install`

### 2. Get Vercel Credentials

After creating the project, you'll need to get the following values:

#### Vercel Token
1. Go to [Vercel Account Settings](https://vercel.com/account/tokens)
2. Create a new token with appropriate permissions
3. Copy the token value

#### Organization ID
1. Go to your Vercel project settings
2. Navigate to the "General" tab
3. Copy the "Organization ID"

#### Project ID
1. In the same project settings page
2. Copy the "Project ID"

### 3. Configure GitHub Secrets

Add the following secrets to your GitHub repository:

1. Go to your GitHub repository
2. Navigate to Settings → Secrets and variables → Actions
3. Add the following repository secrets:

   - `VERCEL_TOKEN`: Your Vercel token from step 2
   - `VERCEL_ORG_ID`: Your organization ID from step 2
   - `VERCEL_PROJECT_ID`: Your project ID from step 2

### 4. Workflow Configuration

The GitHub Action workflow is configured in `.github/workflows/deploy-frontend.yml` and will:

- Trigger on pushes to `main`/`master` branches when frontend files change
- Install dependencies using pnpm
- Build the Next.js application
- Deploy to Vercel

### 5. Manual Deployment

You can also deploy manually by:

1. Pushing changes to the main branch
2. The workflow will automatically trigger
3. Check the Actions tab in GitHub to monitor the deployment

## Troubleshooting

### Common Issues

1. **Build Failures**: Check that all dependencies are properly installed and the build command works locally
2. **Authentication Errors**: Verify that all Vercel secrets are correctly set in GitHub
3. **Path Issues**: Ensure the working directory is set to `./frontend` in the workflow

### Local Testing

Before deploying, test the build locally:

```bash
cd frontend
pnpm install
pnpm build
```

## Environment Variables

If your application requires environment variables:

1. Add them in the Vercel dashboard under Project Settings → Environment Variables
2. Or configure them in the GitHub Action workflow if they're not sensitive

## Custom Domain

To use a custom domain:

1. Go to your Vercel project settings
2. Navigate to the "Domains" tab
3. Add your custom domain
4. Follow the DNS configuration instructions
