# Documentation Deployment Guide

This repository automatically deploys documentation to GitHub Pages with support
for both main branch documentation and PR preview deployments.

## 🚀 Deployment Overview

### Main Branch Documentation

- **URL**: `https://scikit-agent.github.io/scikit-agent/`
- **Trigger**: Pushes to `main` branch or `documentation-*` branches
- **Target**: Root of GitHub Pages site

### PR Preview Documentation

- **URL Pattern**: `https://scikit-agent.github.io/scikit-agent/pr-{number}/`
- **Trigger**: Opening, updating, or reopening pull requests
- **Target**: Subdirectory based on PR number
- **Cleanup**: Automatically removed when PR is closed

## 📋 How It Works

### Workflow: `.github/workflows/docs-deploy.yml`

The deployment workflow consists of several jobs:

1. **`build-docs`**: Builds Sphinx documentation
2. **`deploy-main`**: Deploys to main GitHub Pages (main branch only)
3. **`deploy-pr`**: Deploys PR previews to subdirectories
4. **`cleanup-pr`**: Removes PR previews when PRs are closed

### Automatic Features

- ✅ **PR Comments**: Bot automatically comments with preview links
- ✅ **Conflict Prevention**: Concurrent deployment protection
- ✅ **Auto Cleanup**: PR previews removed when closed
- ✅ **Status Updates**: Comments updated when PRs are closed
- ✅ **Error Handling**: Graceful failure handling

## 🔧 Setup Requirements

### Repository Settings

1. **Enable GitHub Pages**:

   - Go to Settings → Pages
   - Source: "GitHub Actions"
   - No need to select a branch

2. **Workflow Permissions**:
   - Go to Settings → Actions → General
   - Workflow permissions: "Read and write permissions"
   - Check "Allow GitHub Actions to create and approve pull requests"

### Branch Protection

The workflow automatically creates and manages the `gh-pages` branch. No manual
setup required.

## 📱 User Experience

### For PR Authors

1. Open a pull request with documentation changes
2. Wait ~2-3 minutes for deployment
3. Check PR comments for preview link
4. Preview updates automatically on new commits

### For Reviewers

1. Click preview link in PR comments
2. Review documentation changes live
3. Compare with main documentation if needed

### For Maintainers

- Main documentation deploys automatically on merge
- No manual intervention required
- All previews cleaned up automatically

## 🔗 Available URLs

### Primary Documentation

- **GitHub Pages**: https://scikit-agent.github.io/scikit-agent/
- **Read the Docs**: https://scikit-agent.readthedocs.io/

### Development

- **PR Previews**: https://scikit-agent.github.io/scikit-agent/pr-{number}/
- **Branch Docs**: Available for `documentation-*` branches

## 🛠️ Manual Deployment

### Trigger Deployment

```bash
# Manual workflow dispatch
gh workflow run "Deploy Documentation"

# Or push to trigger
git push origin main
```

### Local Testing

```bash
# Build docs locally
pip install ".[docs]"
cd docs
python -m sphinx -b html . _build

# Serve locally
python -m http.server 8000 -d _build
```

## 🐛 Troubleshooting

### Common Issues

1. **Deployment Fails**:

   - Check workflow permissions in repository settings
   - Ensure GitHub Pages is enabled with "GitHub Actions" source

2. **PR Comment Not Created**:

   - Verify workflow has "pull-requests: write" permission
   - Check if bot has proper access rights

3. **Documentation Build Fails**:
   - Sphinx build errors will prevent deployment
   - Check build logs in Actions tab
   - Test locally first

### Debug Steps

1. **Check Actions Tab**: View workflow runs and logs
2. **Verify Permissions**: Ensure proper workflow permissions
3. **Test Locally**: Build documentation locally first
4. **Check Artifacts**: Download build artifacts for debugging

## 📊 Monitoring

### GitHub Actions

- All deployments logged in Actions tab
- Build artifacts available for 30 days
- Deployment status visible in commit checks

### Branch Management

- `gh-pages` branch managed automatically
- PR directories added/removed automatically
- No manual branch management needed

## 🔄 Migration Notes

If migrating from existing documentation setup:

1. **Disable Old Workflows**: Remove conflicting Actions
2. **Update Links**: Update any hardcoded documentation URLs
3. **Test Thoroughly**: Verify both main and PR deployments work
4. **Notify Team**: Update team about new preview URLs

## 📚 Related Documentation

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
