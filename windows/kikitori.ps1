# Registers the scheduled task kikitori: serve.py on SERVE_HOST:SERVE_PORT from .env. Run from a non-elevated
# PowerShell, once, and again after moving the project or changing interpreter; -Force replaces the task. It starts at
# logon and is triggered again every hour, which restarts it if it has exited and is ignored while it runs. S4U logon
# gives it no console, so SERVE_LOG_FILE in .env must be set.
# The task starts python itself, not a shell around it: Stop-ScheduledTask ends only the process the task started,
# and a server under a wrapper outlives it and keeps the port.
$python = (& python -c "import sys; print(sys.executable)")
$serve = Join-Path (Split-Path $PSScriptRoot) "serve.py"

$action = New-ScheduledTaskAction -Execute $python -Argument "`"$serve`""
$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$hourly = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U
Register-ScheduledTask -TaskName kikitori -Action $action -Trigger $logon, $hourly -Settings $settings -Principal $principal -Force | Out-Null
Start-ScheduledTask kikitori
