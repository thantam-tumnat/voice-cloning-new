# compare-snapshot.ps1
# เอา snapshot.zip จากอีกเครื่องมาเทียบกับโค้ดในเครื่องนี้
#
# วิธีใช้ (ไม่ต้องใส่ -Zip ก็ได้ เดี๋ยวมันหา snapshot-*.zip ที่ใหม่สุดให้เอง
# จากทุกไดรฟ์ที่เสียบอยู่ + Desktop + Downloads):
#   powershell -ExecutionPolicy Bypass -File .\tools\compare-snapshot.ps1 -Detail
#
# หรือระบุเองก็ได้:
#   powershell -ExecutionPolicy Bypass -File .\tools\compare-snapshot.ps1 -Zip "E:\snapshot-XXX.zip"
#
# เพิ่ม -Detail เพื่อให้โชว์ diff ทีละบรรทัด (ต้องมี git ในเครื่อง)

param(
    [string]$Zip,
    [string]$Root,
    [switch]$Detail
)

$ErrorActionPreference = 'Stop'
if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath) }

# --- ถ้าไม่ระบุ -Zip: หา snapshot zip ที่ใหม่สุดให้อัตโนมัติ ---
if (-not $Zip) {
    $searchDirs = New-Object System.Collections.Generic.List[string]
    Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name.Length -eq 1 -and $_.Name -ne 'C' } |
        ForEach-Object { $searchDirs.Add($_.Root) }
    $searchDirs.Add([Environment]::GetFolderPath('Desktop'))
    $searchDirs.Add((Join-Path $env:USERPROFILE 'Downloads'))

    $found = @()
    foreach ($d in $searchDirs) {
        if (Test-Path -LiteralPath $d) {
            $found += Get-ChildItem -LiteralPath $d -Filter 'snapshot-*.zip' -File -Recurse -Depth 2 -ErrorAction SilentlyContinue
        }
    }
    if (-not $found) {
        Write-Host "หา snapshot-*.zip ไม่เจอเลย" -ForegroundColor Red
        Write-Host "  - เสียบแฟลชไดรฟ์แล้วยัง?"
        Write-Host "  - หรือระบุเอง:  -Zip ""D:\snapshot-XXXX.zip"""
        Write-Host ""
        Write-Host ("ไดรฟ์ที่มีตอนนี้: " + ((Get-PSDrive -PSProvider FileSystem).Name -join ', '))
        exit 1
    }
    $pick = $found | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $Zip = $pick.FullName
    Write-Host ("เจอ snapshot: {0}  ({1} KB, {2})" -f $Zip, [math]::Round($pick.Length/1KB,1), $pick.LastWriteTime) -ForegroundColor Cyan
    if ($found.Count -gt 1) {
        Write-Host ("  (เจอทั้งหมด {0} ไฟล์ เลือกอันใหม่สุด — ถ้าผิดให้ระบุ -Zip เอง)" -f $found.Count) -ForegroundColor DarkGray
    }
    Write-Host ""
}

$rootFull = (Resolve-Path -LiteralPath $Root).Path
$work = Join-Path $env:TEMP ("cmp_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work -Force | Out-Null
Expand-Archive -LiteralPath (Resolve-Path -LiteralPath $Zip).Path -DestinationPath $work -Force

$otherManifest = Join-Path $work 'MANIFEST.txt'
$otherFiles    = Join-Path $work 'files'
if (-not (Test-Path -LiteralPath $otherManifest)) { throw "ไม่พบ MANIFEST.txt ใน zip — ใช้ zip ที่สร้างจาก make-snapshot.ps1 เท่านั้น" }

Write-Host "=== ข้อมูลเครื่องต้นทาง (INFO.txt) ===" -ForegroundColor Cyan
Get-Content -LiteralPath (Join-Path $work 'INFO.txt') -Encoding UTF8 | ForEach-Object { Write-Host "  $_" }
Write-Host ""

# --- สร้าง manifest ของเครื่องนี้ ด้วยกติกาเดียวกับ make-snapshot.ps1 ---
$dirExclude = '\\(\.git|\.venv|venv|env|__pycache__|node_modules|checkpoints|outputs|voices|\.pytest_cache|\.ruff_cache|\.mypy_cache|\.idea|\.vscode)\\'
$extExclude = @('.wav','.mp3','.flac','.ogg','.pt','.bin','.safetensors','.ckpt','.npy','.npz','.zip','.tar','.gz','.pkl','.pyc')

$mine = @{}
Get-ChildItem -LiteralPath $rootFull -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch $dirExclude -and ($extExclude -notcontains $_.Extension.ToLower()) } |
    ForEach-Object {
        $rel = $_.FullName.Substring($rootFull.Length).TrimStart('\')
        $mine[$rel] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }

$theirs = @{}
Get-Content -LiteralPath $otherManifest -Encoding UTF8 | Where-Object { $_ -match '\S' } | ForEach-Object {
    $h, $rel = $_ -split '  ', 2
    if ($rel) { $theirs[$rel] = $h }
}

$changed  = @()
$onlyMine = @()
$onlyTheirs = @()

foreach ($rel in $mine.Keys) {
    if ($theirs.ContainsKey($rel)) {
        if ($theirs[$rel] -ne $mine[$rel]) { $changed += $rel }
    } else { $onlyMine += $rel }
}
foreach ($rel in $theirs.Keys) {
    if (-not $mine.ContainsKey($rel)) { $onlyTheirs += $rel }
}

Write-Host "=== สรุป ===" -ForegroundColor Cyan
Write-Host ("  เนื้อหาต่างกัน : {0} ไฟล์" -f $changed.Count)     -ForegroundColor Yellow
Write-Host ("  มีแค่เครื่องนี้ : {0} ไฟล์" -f $onlyMine.Count)   -ForegroundColor Green
Write-Host ("  มีแค่อีกเครื่อง : {0} ไฟล์" -f $onlyTheirs.Count) -ForegroundColor Magenta
Write-Host ""

if ($changed.Count)    { Write-Host "--- ต่างกัน ---" -ForegroundColor Yellow;  $changed    | Sort-Object | ForEach-Object { Write-Host "  M  $_" } }
if ($onlyMine.Count)   { Write-Host "--- มีแค่เครื่องนี้ ---" -ForegroundColor Green;   $onlyMine   | Sort-Object | ForEach-Object { Write-Host "  +  $_" } }
if ($onlyTheirs.Count) { Write-Host "--- มีแค่อีกเครื่อง ---" -ForegroundColor Magenta; $onlyTheirs | Sort-Object | ForEach-Object { Write-Host "  -  $_" } }

if ($Detail -and $changed.Count) {
    Write-Host ""
    Write-Host "=== diff ทีละบรรทัด (ซ้าย = อีกเครื่อง, ขวา = เครื่องนี้) ===" -ForegroundColor Cyan
    foreach ($rel in ($changed | Sort-Object)) {
        $a = Join-Path $otherFiles $rel
        $b = Join-Path $rootFull   $rel
        Write-Host ""
        Write-Host ("######## $rel") -ForegroundColor Yellow
        git diff --no-index --no-prefix -- "$a" "$b"
    }
}

Write-Host ""
Write-Host "ไฟล์จากอีกเครื่องแตกไว้ที่: $otherFiles"
Write-Host "(เปิดเทียบเองต่อได้ / ลบทิ้งได้เลยเมื่อเสร็จ)"
