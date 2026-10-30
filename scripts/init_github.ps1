# Production GitHub Initialization and Publishing Script (PowerShell)
# Repository: Sawy05/quantum-hybrid-portfolio-engine

$ErrorActionPreference = "Continue"

$RepoName = "quantum-hybrid-portfolio-engine"
$GitHubUser = "Sawy05"
$RemoteUrl = "https://github.com/$GitHubUser/$RepoName.git"

Write-Host "=== Initializing Quantum Hybrid Portfolio Engine Repository ===" -ForegroundColor Cyan

# 1. Check Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "[-] git is not installed or not in PATH."
    exit 1
}
Write-Host "[+] Git detected: $(git --version)" -ForegroundColor Green

# 2. Check GitHub CLI (gh)
$hasGh = Get-Command gh -ErrorAction SilentlyContinue
$ghAuth = $false
if ($hasGh) {
    Write-Host "[+] GitHub CLI detected." -ForegroundColor Green
    $authCheck = gh auth status 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[+] GitHub CLI is authenticated." -ForegroundColor Green
        $ghAuth = $true
    } else {
        Write-Host "[!] GitHub CLI installed but not authenticated." -ForegroundColor Yellow
    }
} else {
    Write-Host "[!] GitHub CLI (gh) not found in PATH. Proceeding with standard git remote." -ForegroundColor Yellow
}

# 3. Initialize Git Repository
if (-not (Test-Path ".git")) {
    Write-Host "[*] Initializing new git repository..." -ForegroundColor Cyan
    git init -b main
} else {
    Write-Host "[*] Git repository already initialized." -ForegroundColor Cyan
    git checkout -B main
}

# 4. Configure Git User if not set
$userName = git config user.name
if (-not $userName) {
    git config user.name "Ahmed Elsawy"
}
$userEmail = git config user.email
if (-not $userEmail) {
    git config user.email "Sawy05@users.noreply.github.com"
}

# 5. Stage files and Commit
Write-Host "[*] Staging files..." -ForegroundColor Cyan
git add .
git commit -m "feat: initial production-grade hybrid quantum-classical financial regime & QAOA portfolio engine"

# 6. Remote Repository Setup
if ($ghAuth) {
    Write-Host "[*] Verifying/creating GitHub repository via GitHub CLI..." -ForegroundColor Cyan
    gh repo create "$GitHubUser/$RepoName" --public --description "Production-Grade Hybrid Quantum-Classical Financial Regime Detection & QAOA Portfolio Optimization Engine" --source=. --remote=origin --push
} else {
    Write-Host "[*] Configuring remote origin to $RemoteUrl..." -ForegroundColor Cyan
    $remotes = git remote
    if ($remotes -contains "origin") {
        git remote set-url origin $RemoteUrl
    } else {
        git remote add origin $RemoteUrl
    }
    Write-Host "[*] Pushing to main branch..." -ForegroundColor Cyan
    git push -u origin main
}

Write-Host "[+] Synchronization script complete." -ForegroundColor Green
