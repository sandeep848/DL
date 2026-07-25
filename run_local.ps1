# run_local.ps1
# PowerShell script to run Ghibli LoRA training and evaluation locally on CPU.

$PythonPath = "C:\Users\Sandip\.vscode\anaconda\envs\CV\python.exe"

Write-Host "==============================================" -ForegroundColor Green
Write-Host "🎨 Ghibli Market: Local CPU Execution Runner" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green

# Step 1: Run Training
Write-Host "`n[Step 1/2] Starting LoRA Training on CPU..." -ForegroundColor Cyan
Write-Host "Note: Training configured for 20 steps for rapid validation." -ForegroundColor Yellow
Write-Host "Command: python code/train_lora.py --data_dir style_imgs/512 --instance_token '<sks>' --output_dir lora_out --rank 8 --max_steps 20 --overwrite --no_augmentation" -ForegroundColor Gray

& $PythonPath code/train_lora.py `
  --data_dir style_imgs/512 `
  --instance_token "<sks>" `
  --output_dir lora_out `
  --rank 8 `
  --max_steps 20 `
  --overwrite `
  --no_augmentation

if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed! Exiting."
    Exit $LASTEXITCODE
}

# Step 2: Run Evaluation
Write-Host "`n[Step 2/2] Starting LoRA Evaluation on CPU..." -ForegroundColor Cyan
Write-Host "Command: python code/eval_lora.py --weights lora_out/pytorch_lora_weights.safetensors --prompt 'a busy market, in <sks> style' --outdir samples" -ForegroundColor Gray

& $PythonPath code/eval_lora.py `
  --weights lora_out/pytorch_lora_weights.safetensors `
  --prompt "a busy market, in <sks> style" `
  --outdir samples

if ($LASTEXITCODE -ne 0) {
    Write-Error "Evaluation failed! Exiting."
    Exit $LASTEXITCODE
}

Write-Host "`n==============================================" -ForegroundColor Green
Write-Host "✅ Local execution completed successfully!" -ForegroundColor Green
Write-Host "Outputs saved to:" -ForegroundColor Green
Write-Host "  - Weights: lora_out/pytorch_lora_weights.safetensors" -ForegroundColor Green
Write-Host "  - Samples: samples/" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
