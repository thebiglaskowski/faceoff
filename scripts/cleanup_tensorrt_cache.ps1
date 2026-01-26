# TensorRT Cache Cleanup Script
# Run this when the FaceOff app is NOT running

Write-Host "`n🧹 Cleaning up duplicate TensorRT cache..." -ForegroundColor Cyan

$oldCache = "tensorrt_cache"
$newCache = "cache\tensorrt"

if (Test-Path $oldCache) {
    Write-Host "`nRemoving old cache: $oldCache" -ForegroundColor Yellow
    
    # Stop any processes that might be using the files
    $files = Get-ChildItem $oldCache -Recurse -File
    Write-Host "Found $($files.Count) cache files to remove"
    
    try {
        Remove-Item $oldCache -Recurse -Force -ErrorAction Stop
        Write-Host "✅ Successfully removed $oldCache" -ForegroundColor Green
    }
    catch {
        Write-Host "⚠️  Could not delete (files may be in use)" -ForegroundColor Yellow
        Write-Host "   Please close FaceOff and run this script again" -ForegroundColor Yellow
        Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "✅ Already clean - $oldCache doesn't exist" -ForegroundColor Green
}

Write-Host "`n📊 Current cache structure:" -ForegroundColor Cyan
if (Test-Path $newCache) {
    $cacheSize = (Get-ChildItem $newCache -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "   ✅ $newCache exists ($([math]::Round($cacheSize, 2)) MB)" -ForegroundColor Green
}
else {
    Write-Host "   ⚠️  $newCache doesn't exist yet (will be created on first use)" -ForegroundColor Yellow
}

Write-Host "`n✅ Cleanup complete!`n" -ForegroundColor Green
