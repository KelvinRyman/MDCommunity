# 相当于 bash 的 set -e (遇到错误立即停止)
$ErrorActionPreference = 'Stop'

# 相当于 bash 的 set -x (打印执行的命令)，如果觉得输出太乱可以注释掉下面这行
# Set-PSDebug -Trace 1

# 参数处理 logic
# if [ $# -eq 2 ]; then
if ($args.Count -eq 2) {
    # 获取传入的参数
    $address_dirtory = $args[0]
    $input_filename = $args[1]
} else {
    # 使用默认值
    $address_dirtory = 'MultiDismantler_unit_cost'
    $input_filename = 'train'
}

# 主逻辑判断
if ($address_dirtory -eq "MultiDismantler_degree_cost") {
    
    if ($input_filename -eq "train") {
        # Training
        python -u .\MultiDismantler_degree_cost\train.py
    } elseif ($input_filename -eq "testReal") {
        python -u .\MultiDismantler_degree_cost\testReal.py --output "..\..\results\degreecost\MultiDismantler_real"
    } elseif ($input_filename -eq "testSynthetic") {
        python -u .\MultiDismantler_degree_cost\testSynthetic.py --output "..\..\results\degreecost\MultiDismantler_syn\"
    } elseif ($input_filename -eq "drawLmcc") {
        python -u .\MultiDismantler_degree_cost\drawWeight.py --output "..\..\results\degreecost\MultiDismantler_audc\"
    }

} elseif ($address_dirtory -eq "MultiDismantler_unit_cost") {

    if ($input_filename -eq "train") {
        # Training
        python -u .\MultiDismantler_unit_cost\train.py
    } elseif ($input_filename -eq "testReal") {
        python -u .\MultiDismantler_unit_cost\testReal.py --output "..\..\results\unitcost\MultiDismantler_real"
    } elseif ($input_filename -eq "testSynthetic") {
        python -u .\MultiDismantler_unit_cost\testSynthetic.py --output "..\..\results\unitcost\MultiDismantler_syn\"
    } elseif ($input_filename -eq "drawLmcc") {
        python -u .\MultiDismantler_unit_cost\drawUnweight.py --output "..\results\unitcost\MultiDismantler_audc\"
    }

} elseif ($address_dirtory -eq "MultiDismantler_community") {

    if ($input_filename -eq "train") {
        # Training
        python -u .\MultiDismantler_community\train.py
    } elseif ($input_filename -eq "testReal") {
        python -u .\MultiDismantler_community\testReal.py --output "..\..\results\community\MultiDismantler_real"
    } elseif ($input_filename -eq "testSynthetic") {
        python -u .\MultiDismantler_community\testSynthetic.py --output "..\..\results\community\MultiDismantler_syn\"
    } elseif ($input_filename -eq "drawLmcc") {
        python -u .\MultiDismantler_community\drawCommunity.py --output "..\..\results\community\MultiDismantler_audc\"
    }

} else {
    Write-Host "No training or testing will be performed!"
}