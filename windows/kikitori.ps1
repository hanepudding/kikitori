# Registers the scheduled task kikitori: serve.py on SERVE_HOST:SERVE_PORT from .env. Run from a non-elevated
# PowerShell, once, and again after moving the project or changing interpreter; -Force replaces the task. It starts at
# logon and is triggered again every hour, which restarts it if it has exited and is ignored while it runs. S4U logon
# gives it no console window, so cmd redirects the output into the log.
# -X utf8: redirected output otherwise takes the ANSI code page, which cannot encode every Chinese character, and a
# traceback that fails to print kills the job thread with it.
$python = (& python -c "import sys; print(sys.executable)")
$serve = Join-Path (Split-Path $PSScriptRoot) "serve.py"
$log = "$env:USERPROFILE\.local\state\kikitori\serve.log"

New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"`"$python`" -X utf8 -u `"$serve`" >> `"$log`" 2>&1`""
$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$hourly = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U
Register-ScheduledTask -TaskName kikitori -Action $action -Trigger $logon, $hourly -Settings $settings -Principal $principal -Force | Out-Null
Start-ScheduledTask kikitori
