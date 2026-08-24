# make-snapshot.ps1
# เก็บ "snapshot" ของไฟล์โค้ดในโปรเจค (ตัดพวก .venv / checkpoints / ไฟล์เสียง / โมเดล ออก)
# ออกมาเป็นไฟล์ zip ไฟล์เดียวไว้ที่ Desktop -> ก็อปใส่แฟลชไดรฟ์ไปเทียบอีกเครื่องได้เลย
#
# วิธีใช้ (รันในโฟลเดอร์โปรเจคเครื่องไหนก็ได้):
#   powershell -ExecutionPolicy Bypass -File .\tools\make-snapshot.ps1

param(
    [string]$Root,
    [string]$OutDir = [Environment]::GetFolderPath('Desktop')
)

$ErrorActionPreference = 'Stop'
if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath) }

# โฟลเดอร์ที่ไม่ต้องเทียบ
$dirExclude = '\\(\.git|\.venv|venv|env|__pycache__|node_modules|checkpoints|outputs|voices|\.pytest_cache|\.ruff_cache|\.mypy_cache|\.idea|\.vscode)\\'
# นามสกุลที่ไม่ต้องเทียบ (ไฟล์ใหญ่ / binary)
$extExclude = @('.wav','.mp3','.flac','.ogg','.pt','.bin','.safetensors','.ckpt','.npy','.npz','.zip','.tar','.gz','.pkl','.pyc')

$rootFull = (Resolve-Path -LiteralPath $Root).Path
Write-Host "Root: $rootFull"

$stage = Join-Path $env:TEMP ("snap_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stage -Force | Out-Null

$files = Get-ChildItem -LiteralPath $rootFull -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch $dirExclude -and ($extExclude -notcontains $_.Extension.ToLower()) }

$lines = New-Object System.Collections.Generic.List[string]
foreach ($f in $files) {
    $rel  = $f.FullName.Substring($rootFull.Length).TrimStart('\')
    $hash = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash
    $lines.Add(("{0}  {1}" -f $hash, $rel))

    $dest = Join-Path (Join-Path $stage 'files') $rel
    New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
    Copy-Item -LiteralPath $f.FullName -Destination $dest -Force
}

($lines | Sort-Object) | Out-File -FilePath (Join-Path $stage 'MANIFEST.txt') -Encoding utf8

# ข้อมูลประกอบ: เครื่องไหน / เวลาไหน / git อยู่ commit ไหน
$info = New-Object System.Collections.Generic.List[string]
$info.Add("computer   : $env:COMPUTERNAME")
$info.Add("user       : $env:USERNAME")
$info.Add("root       : $rootFull")
$info.Add("taken_at   : " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
$info.Add("file_count : " + $files.Count)
Get-ChildItem -LiteralPath $rootFull -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq '.git' } | ForEach-Object {
        $repo = Split-Path -Parent $_.FullName
        $info.Add("")
        $info.Add("git repo   : " + $repo.Substring($rootFull.Length).TrimStart('\'))
        $info.Add("  HEAD     : " + (git -C $repo log -1 --format='%h %ad %s' --date=short 2>$null))
        $info.Add("  branch   : " + (git -C $repo rev-parse --abbrev-ref HEAD 2>$null))
        $info.Add("  dirty    :")
        (git -C $repo status --porcelain 2>$null) | ForEach-Object { $info.Add("    $_") }
    }
$info | Out-File -FilePath (Join-Path $stage 'INFO.txt') -Encoding utf8

$zip = Join-Path $OutDir ("snapshot-{0}-{1}.zip" -f $env:COMPUTERNAME, (Get-Date -Format 'yyyyMMdd-HHmm'))
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
Remove-Item -LiteralPath $stage -Recurse -Force

Write-Host ""
Write-Host "เสร็จแล้ว: $zip"
Write-Host ("ขนาด: {0} KB / {1} ไฟล์" -f [math]::Round((Get-Item -LiteralPath $zip).Length/1KB,1), $files.Count)
