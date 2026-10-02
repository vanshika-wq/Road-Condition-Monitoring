Set-Location "C:\Users\juhis\OneDrive\Desktop\capstone"

git add .

if (-not (git diff --cached --quiet)) {
    git commit -m "Daily progress - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    git push origin main
}
