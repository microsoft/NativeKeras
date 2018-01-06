$location = Get-Location

$soluctionDir = Split-Path $PSScriptRoot

$path = Join-Path $soluctionDir "\KerasProtoLib"
Set-Location $path

$command = "$env:PROTOC_DIR" + "protoc.exe" + " --cpp_out=. --proto_path ..\Proto ..\Proto\KerasProto.proto"
Invoke-Expression $command

$path = Join-Path $soluctionDir "\NativeKeras"
Set-Location $path

$command = "$env:PROTOC_DIR" + "protoc.exe" + " --csharp_out=. --proto_path ..\Proto ..\Proto\KerasProto.proto"
Invoke-Expression $command

Set-Location $location
