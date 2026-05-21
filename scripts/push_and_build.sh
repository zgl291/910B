#!/bin/bash
# =============================================================================
# Push to GitHub and trigger CI build
# Usage: bash scripts/push_and_build.sh <github-username> <repo-name>
# =============================================================================
set -euo pipefail

GH_USER="${1:?Usage: bash push_and_build.sh <github-user> <repo-name>}"
REPO_NAME="${2:?Usage: bash push_and_build.sh <github-user> <repo-name>}"
REMOTE_URL="https://github.com/${GH_USER}/${REPO_NAME}.git"

echo "============================================================"
echo "  Push to GitHub and Trigger CI Build"
echo "============================================================"
echo "  User:  ${GH_USER}"
echo "  Repo:  ${REPO_NAME}"
echo "  URL:   ${REMOTE_URL}"
echo "============================================================"

# Check if remote already exists
CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
if [ -n "${CURRENT_REMOTE}" ]; then
    echo "Existing remote: ${CURRENT_REMOTE}"
    read -p "Replace remote origin? (y/N): " REPLACE
    if [[ "${REPLACE}" =~ ^[Yy] ]]; then
        git remote set-url origin "${REMOTE_URL}"
    else
        echo "Aborted."
        exit 1
    fi
else
    git remote add origin "${REMOTE_URL}"
fi

# Check if repo exists on GitHub
echo ""
echo "Checking if repository exists on GitHub..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://api.github.com/repos/${GH_USER}/${REPO_NAME}")

if [ "${HTTP_CODE}" = "404" ]; then
    echo "Repository not found on GitHub."
    echo ""
    echo "Please create the repository first:"
    echo "  1. Go to: https://github.com/new"
    echo "  2. Repository name: ${REPO_NAME}"
    echo "  3. DO NOT initialize with README/.gitignore"
    echo "  4. Click 'Create repository'"
    echo ""
    read -p "Press Enter after creating the repository..."
elif [ "${HTTP_CODE}" = "200" ]; then
    echo "[OK] Repository exists on GitHub."
else
    echo "[WARN] HTTP ${HTTP_CODE} - may need authentication."
    echo "  Make sure you have 'gh' CLI authenticated or create a PAT."
fi

# Commit and push
echo ""
echo "Committing and pushing to GitHub..."

git add -A
git commit -m "chore: init ascend 910b remote sensing container environment

- Dockerfile for production (ARM64 + CANN + torch_npu)
- All Python dependencies (50+ packages)
- Migration tools (code adapter, model converter, distributed)
- GitHub Actions CI for automated build

Co-Authored-By: Claude <noreply@anthropic.com>" || echo "No changes to commit."

git push -u origin HEAD 2>&1 || {
    echo ""
    echo "Push failed. Possible reasons:"
    echo "  1. Not authenticated. Run: gh auth login"
    echo "  2. Or use PAT: git push https://<TOKEN>@github.com/${GH_USER}/${REPO_NAME}.git HEAD:main"
    echo ""
    echo "Alternative: Create the repo on GitHub web, then:"
    echo "  git push -u origin main"
    exit 1
}

echo ""
echo "============================================================"
echo "  Push Complete!"
echo "============================================================"
echo ""
echo "GitHub Actions build has been triggered."
echo "Monitor progress at:"
echo "  https://github.com/${GH_USER}/${REPO_NAME}/actions"
echo ""
echo "After build completes (~30-60 min), pull the image:"
echo "  bash scripts/pull_image.sh"
echo ""
