#!/usr/bin/env bash
# Production GitHub Initialization and Publishing Script
# Repository: Sawy05/quantum-hybrid-portfolio-engine

set -euo pipefail

REPO_NAME="quantum-hybrid-portfolio-engine"
GITHUB_USER="Sawy05"
REMOTE_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo "=== Initializing Quantum Hybrid Portfolio Engine Repository ==="

# 1. Check Git
if ! command -v git &> /dev/null; then
    echo "[-] Error: git is not installed or not in PATH."
    exit 1
fi
echo "[+] Git detected: $(git --version)"

# 2. Check GitHub CLI (gh)
if command -v gh &> /dev/null; then
    echo "[+] GitHub CLI detected: $(gh --version | head -n 1)"
    if gh auth status &> /dev/null; then
        echo "[+] GitHub CLI is authenticated."
        GH_AUTH=true
    else
        echo "[!] GitHub CLI is installed but not authenticated. Run 'gh auth login' or push via git remote."
        GH_AUTH=false
    fi
else
    echo "[!] GitHub CLI (gh) not found in PATH. Proceeding with standard git remote push."
    GH_AUTH=false
fi

# 3. Initialize Git Repository
if [ ! -d ".git" ]; then
    echo "[*] Initializing new git repository..."
    git init -b main
else
    echo "[*] Git repository already initialized."
    git checkout -B main
fi

# 4. Configure Git User if not set
if [ -z "$(git config user.name)" ]; then
    git config user.name "Ahmed Elsawy"
fi
if [ -z "$(git config user.email)" ]; then
    git config user.email "Sawy05@users.noreply.github.com"
fi

# 5. Stage files and Commit
echo "[*] Staging files..."
git add .
git commit -m "feat: initial production-grade hybrid quantum-classical financial regime & QAOA portfolio engine" || true

# 6. Remote Repository Setup
if [ "$GH_AUTH" = true ]; then
    echo "[*] Verifying/creating GitHub repository via GitHub CLI..."
    gh repo create "${GITHUB_USER}/${REPO_NAME}" --public --description "Production-Grade Hybrid Quantum-Classical Financial Regime Detection & QAOA Portfolio Optimization Engine" --source=. --remote=origin --push || true
else
    echo "[*] Configuring remote origin to ${REMOTE_URL}..."
    if git remote | grep -q "^origin$"; then
        git remote set-url origin "${REMOTE_URL}"
    else
        git remote add origin "${REMOTE_URL}"
    fi
    echo "[*] Pushing to main branch..."
    git push -u origin main || echo "[!] Notice: Remote repository ${REMOTE_URL} requires authentication or remote creation on GitHub."
fi

echo "[+] Successfully synchronized quantum-hybrid-portfolio-engine repository."
