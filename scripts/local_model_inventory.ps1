$roots = @(
  "G:/More-models",
  "H:/OLLAMA-Models/GGUF"
)

$patterns = @(
  "*Qwen3-4B-Q8_0*",
  "*Qwen3.5-9B-Q4_K_M*",
  "*Qwen3-8B-Q4_K_M*",
  "*Qwen3-4B-Instruct-2507-Q6_K*",
  "*Qwen3-MOE-4x0.6B*",
  "*Qwen3.5-0.8B*",
  "*SmolLM2-135M*",
  "*Qwen3-30B-A3B*",
  "*laguna*",
  "*Ornith*",
  "*Tiel*",
  "*omnicoder*",
  "*Qwen2.5-Coder*",
  "*Phi-3.5*",
  "*smolcode*",
  "*Samastam*",
  "*VibeThinker*",
  "*qwen3vl*",
  "*Spark_one*",
  "*lacuna*",
  "*Q3.5-9B*",
  "*qwable*",
  "*nvfp4*",
  "*qwen3-1.7b*",
  "*mythos*",
  "*.gguf"
)

$seen = @{}
foreach ($root in $roots) {
  if (-not (Test-Path $root)) {
    Write-Output "ROOT MISSING: $root"
    continue
  }
  Write-Output "ROOT EXISTS: $root"
  Get-ChildItem -LiteralPath $root -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '\.gguf$' } |
    ForEach-Object {
      foreach ($pat in $patterns) {
        if ($_.Name -like $pat) {
          $key = $_.FullName
          if (-not $seen.ContainsKey($key)) {
            $seen[$key] = 1
            $mb = [math]::Round($_.Length/1MB, 2)
            Write-Output "$mb MB | $($_.FullName)"
          }
        }
      }
    }
}
Write-Output "---ALL GGUF COUNT---"
$found = 0
foreach ($root in $roots) {
  if (-not (Test-Path $root)) { continue }
  $found += (Get-ChildItem -LiteralPath $root -File -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '\.gguf$' }).Count
}
Write-Output "$found gguf files found across searched roots"
