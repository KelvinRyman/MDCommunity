# Activate virtual environment outside or uncomment below
# .venv\Scripts\Activate.ps1

Write-Host "=== 开始训练基线模型 (Unit Cost) ==="
python -u MultiDismantler_unit_cost/train.py
Write-Host "基线模型训练完成。"

Write-Host "=== 开始训练社区模型 (Community Aware) ==="
python -u MultiDismantler_community/train.py
Write-Host "社区模型训练完成。"

Write-Host "=== 所有训练任务结束 ==="
