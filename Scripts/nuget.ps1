$location = Get-Location

$soluctionDir = Split-Path $PSScriptRoot

$path = Join-Path $soluctionDir "\NativeKeras"
Set-Location $path

$exePath = Get-Command nuget.exe | Select-Object -ExpandProperty Definition
$args = "pack NativeKeras.nuspec -Prop Configuration=Release"
Start-Process -Wait -NoNewWindow -ArgumentList $args $exePath

Set-Location $location
