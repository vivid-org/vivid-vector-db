# Vivid CLI Integration Tests
# Exercises every command for both HNSW and Flat index types

$VIVID = "$PSScriptRoot\..\target\release\vivid-cli.exe"
$TMP = "$PSScriptRoot\..\target\tmp-test"
$PASS = 0
$FAIL = 0

# Clean slate
if (Test-Path -LiteralPath $TMP) { Remove-Item -Path "$TMP\*" -Force; Remove-Item -LiteralPath $TMP -Force }
New-Item -ItemType Directory -Path $TMP | Out-Null

$HNSW = "$TMP\test.hnsw"
$FLAT = "$TMP\test.flat"
$NONEXIST = "$TMP\nonexistent.vidx"

function ok { $script:PASS += 1; Write-Host "  PASS" -ForegroundColor Green }
function fail { $script:FAIL += 1; Write-Host "  FAIL: $args" -ForegroundColor Red }

function assert-contains {
    param($Haystack, $Needle)
    if ($LASTEXITCODE -ne 0) { return fail "exit code $LASTEXITCODE" }
    if ($Haystack -match $Needle) { ok } else { fail "expected '$Needle' in '$Haystack'" }
}
function assert-not-contains {
    param($Haystack, $Needle)
    if ($LASTEXITCODE -ne 0) { return fail "exit code $LASTEXITCODE" }
    if ($Haystack -notmatch $Needle) { ok } else { fail "unexpected '$Needle' in '$Haystack'" }
}
function assert-code {
    param($Expected)
    if ($LASTEXITCODE -eq $Expected) { ok } else { fail "expected exit code $Expected, got $LASTEXITCODE" }
}

# ============================================================
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  HNSW Index Tests" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# 1. Create
Write-Host "1) Create HNSW index"
$r = & $VIVID create -i $HNSW -d 3
assert-contains $r "Created empty HNSW"

# 2. Info (empty)
Write-Host "2) Info (empty HNSW)"
$r = & $VIVID info -i $HNSW
assert-contains $r "HNSW"
assert-contains $r "Vectors:\s*0"

# 3. Insert vectors
Write-Host "3) Insert v1"
$r = & $VIVID insert -i $HNSW -n 1 -v "[1.0, 0.0, 0.0]"
assert-contains $r "Inserted ID 1"

Write-Host "4) Insert v2"
$r = & $VIVID insert -i $HNSW -n 2 -v "[0.0, 1.0, 0.0]"
assert-contains $r "Inserted ID 2"

Write-Host "5) Insert v3"
$r = & $VIVID insert -i $HNSW -n 3 -v "[0.0, 0.0, 1.0]"
assert-contains $r "Inserted ID 3"

# 4. Duplicate insert rejected
Write-Host "6) Duplicate insert rejected"
$r = & $VIVID insert -i $HNSW -n 1 -v "[0.5, 0.5, 0.0]" 2>&1
if ($LASTEXITCODE -ne 0 -and $r -match "DuplicateId") { ok } else { fail "expected DuplicateId error, got exit $LASTEXITCODE : $r" }

# 5. Get
Write-Host "7) Get v1"
$r = & $VIVID get -i $HNSW -n 1
assert-contains $r "ID 1"
assert-contains $r "1.0.*0.0.*0.0"

Write-Host "8) Get nonexistent"
$r = & $VIVID get -i $HNSW -n 999 2>&1
if ($LASTEXITCODE -eq 0 -and $r -match "not found") { ok } else { fail "expected 'not found'" }

# 6. Search
Write-Host "9) Search"
$r = & $VIVID search -i $HNSW -q "[0.9, 0.1, 0.0]" -k 2
assert-contains $r "Search top 2 via HNSW"
assert-contains $r "ID: 1"

# 7. Update
Write-Host "10) Update v1"
$r = & $VIVID update -i $HNSW -n 1 -v "[0.5, 0.5, 0.0]"
assert-contains $r "Updated ID 1"

Write-Host "11) Verify updated vector"
$r = & $VIVID get -i $HNSW -n 1
assert-contains $r "0.5.*0.5.*0.0"

Write-Host "12) Update nonexistent"
$r = & $VIVID update -i $HNSW -n 999 -v "[0.0, 0.0, 0.0]" 2>&1
if ($LASTEXITCODE -ne 0 -and $r -match "IdNotFound") { ok } else { fail "expected IdNotFound error, got exit $LASTEXITCODE : $r" }

# 8. Upsert (new)
Write-Host "13) Upsert new vector (ID 4)"
$r = & $VIVID upsert -i $HNSW -n 4 -v "[0.1, 0.2, 0.3]"
assert-contains $r "Inserted ID 4"

# 9. Upsert (existing)
Write-Host "14) Upsert existing (ID 4)"
$r = & $VIVID upsert -i $HNSW -n 4 -v "[0.9, 0.8, 0.7]"
assert-contains $r "Updated ID 4"

# 10. Info
Write-Host "15) Info after mutations"
$r = & $VIVID info -i $HNSW
assert-contains $r "HNSW"
assert-contains $r "Vectors:\s*4"

# 11. Delete
Write-Host "16) Delete ID 3"
$r = & $VIVID delete -i $HNSW -n 3
assert-contains $r "Deleted ID 3"

Write-Host "17) Verify deletion"
$r = & $VIVID get -i $HNSW -n 3 2>&1
if ($LASTEXITCODE -eq 0 -and $r -match "not found") { ok } else { fail "expected 'not found'" }

Write-Host "18) Delete nonexistent"
$r = & $VIVID delete -i $HNSW -n 999 2>&1
if ($LASTEXITCODE -ne 0 -and $r -match "IdNotFound") { ok } else { fail "expected IdNotFound error, got exit $LASTEXITCODE : $r" }

# 12. Persistence: reload and verify count
Write-Host "19) Persistence check"
$r = & $VIVID info -i $HNSW
assert-contains $r "Vectors:\s*3"

# ============================================================
Write-Host "" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Flat Index Tests" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# 13. Create flat
Write-Host "20) Create Flat index"
$r = & $VIVID create -i $FLAT -d 3 -t flat
assert-contains $r "Created empty Flat"

# 14. Info (empty flat)
Write-Host "21) Info (empty Flat)"
$r = & $VIVID info -i $FLAT
assert-contains $r "Flat"
assert-contains $r "Vectors:\s*0"

# 15. Insert
Write-Host "22) Insert v1 (flat)"
$r = & $VIVID insert -i $FLAT -n 10 -v "[1.0, 0.0, 0.0]"
assert-contains $r "Inserted ID 10"

Write-Host "23) Insert v2 (flat)"
$r = & $VIVID insert -i $FLAT -n 20 -v "[0.0, 1.0, 0.0]"
assert-contains $r "Inserted ID 20"

# 16. Duplicate insert rejected (flat)
Write-Host "24) Duplicate insert rejected (flat)"
$r = & $VIVID insert -i $FLAT -n 10 -v "[0.5, 0.5, 0.0]" 2>&1
if ($LASTEXITCODE -ne 0 -and $r -match "DuplicateId") { ok } else { fail "expected DuplicateId error, got exit $LASTEXITCODE : $r" }

# 17. Get
Write-Host "25) Get (flat)"
$r = & $VIVID get -i $FLAT -n 10
assert-contains $r "ID 10"
assert-contains $r "1.0.*0.0.*0.0"

# 18. Search (flat)
Write-Host "26) Search (flat)"
$r = & $VIVID search -i $FLAT -q "[0.9, 0.1, 0.0]" -k 2
assert-contains $r "Search top 2 via Flat"
assert-contains $r "ID: 10"

# 19. Update (flat)
Write-Host "27) Update (flat)"
$r = & $VIVID update -i $FLAT -n 10 -v "[0.5, 0.5, 0.0]"
assert-contains $r "Updated ID 10"

Write-Host "28) Verify updated (flat)"
$r = & $VIVID get -i $FLAT -n 10
assert-contains $r "0.5.*0.5.*0.0"

# 20. Delete (flat)
Write-Host "29) Delete (flat)"
$r = & $VIVID delete -i $FLAT -n 20
assert-contains $r "Deleted ID 20"

Write-Host "30) Verify deletion (flat)"
$r = & $VIVID get -i $FLAT -n 20 2>&1
if ($LASTEXITCODE -eq 0 -and $r -match "not found") { ok } else { fail "expected 'not found'" }

# ============================================================
Write-Host "" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Upsert from scratch test" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

Write-Host "31) Upsert creates new index if missing"
$r = & $VIVID upsert -i $NONEXIST -n 1 -v "[1.0,2.0,3.0]"
assert-contains $r "Creating new HNSW index"

Write-Host "32) Upsert into existing (created above)"
$r = & $VIVID upsert -i $NONEXIST -n 2 -v "[4.0,5.0,6.0]"
assert-contains $r "Inserted ID 2"

Write-Host "33) Verify both vectors survive"
$r = & $VIVID info -i $NONEXIST
assert-contains $r "Vectors:\s*2"

# ============================================================
Write-Host "" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Error path tests" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

Write-Host "34) Error: operation on nonexistent file"
$r = & $VIVID info -i "$TMP\no-such-file.vidx" 2>&1
if ($LASTEXITCODE -ne 0) { ok } else { fail "expected nonzero exit" }

Write-Host "35) Error: invalid vector JSON"
$r = & $VIVID insert -i $HNSW -n 99 -v "not-json" 2>&1
if ($LASTEXITCODE -ne 0) { ok } else { fail "expected nonzero exit" }

Write-Host "36) Create with --force overwrite"
$r = & $VIVID create -i $HNSW -d 5 -f
assert-contains $r "Created empty HNSW"

Write-Host "37) Verify dimension changed (5)"
$r = & $VIVID info -i $HNSW
assert-contains $r "Dimension: 5"

# ============================================================
Write-Host "" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Batch Insert Tests" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# 38. Batch insert into HNSW
Write-Host "38) Batch insert into HNSW"
# Recreate with dim=3 for batch insert (was overwritten to dim=5 in test 36)
& $VIVID create -i $HNSW -d 3 -f | Out-Null
$json = "$TMP\batch_hnsw.json"
[System.IO.File]::WriteAllText($json, '[[1, [0.1, 0.2, 0.3]], [2, [0.4, 0.5, 0.6]], [3, [0.7, 0.8, 0.9]]]')
$r = & $VIVID batch-insert -i $HNSW -f $json
assert-contains $r "Inserted 3 vectors"

Write-Host "39) Verify batch via get"
$r = & $VIVID get -i $HNSW -n 2
assert-contains $r "0.4.*0.5.*0.6"

Write-Host "40) Batch insert into Flat"
& $VIVID create -i $FLAT -d 3 -t flat -f | Out-Null
$json2 = "$TMP\batch_flat.json"
[System.IO.File]::WriteAllText($json2, '[[10, [1.0, 0.0, 0.0]], [20, [0.0, 1.0, 0.0]]]')
$r = & $VIVID batch-insert -i $FLAT -f $json2
assert-contains $r "Inserted 2 vectors"

Write-Host "41) Verify flat batch via get"
$r = & $VIVID get -i $FLAT -n 20
assert-contains $r "0.0.*1.0.*0.0"

# ============================================================
Write-Host "" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host "  Results: $PASS passed, $FAIL failed"
Write-Host "=============================" -ForegroundColor Cyan

# Cleanup
Remove-Item -Path "$TMP\*" -Force; Remove-Item -LiteralPath $TMP -Force

if ($FAIL -gt 0) { exit 1 } else { exit 0 }
