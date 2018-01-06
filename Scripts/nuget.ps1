$location = Get-Location

$soluctionDir = Split-Path $PSScriptRoot

$path = Join-Path $soluctionDir "\NativeKeras"
Set-Location $path

$command = "nuget.exe pack NativeKeras.nuspec -Prop Configuration=Release"
Invoke-Expression $command

Set-Location $location
