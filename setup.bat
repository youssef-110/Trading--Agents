@echo off
setlocal
title Ultron trader setup
echo.
echo  Setting up Ultron trader...
echo.

set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
set "DEST=%USERPROFILE%\Projects\ultron-trader"

rem ---- 1. check tools -------------------------------------------------
where python >nul 2>nul || (
  echo  [!] Python not found. Install it from https://www.python.org/downloads/
  echo      and tick "Add python.exe to PATH" during install. Then run this again.
  goto :end
)
where git >nul 2>nul || (
  echo  [!] Git not found. Install it from https://git-scm.com/download/win
  echo      Claude Code on Windows needs it too. Then run this again.
  goto :end
)

rem ---- 2. copy the project into Projects ------------------------------
if /i "%SRC%"=="%DEST%" goto :inplace
if exist "%DEST%\main.py" (
  echo  [i] %DEST% already exists, leaving it as is and using it.
) else (
  echo  [1/5] Copying project to %DEST%
  robocopy "%SRC%" "%DEST%" /E /XD .venv __pycache__ .pytest_cache .git /XF setup.log >nul
)
:inplace
cd /d "%DEST%" || goto :end

rem VS Code settings ship in vscode-config because .vscode can't be written remotely
if exist "vscode-config\settings.json" (
  if not exist ".vscode" mkdir ".vscode"
  copy /y "vscode-config\*.json" ".vscode\" >nul
)

rem ---- 3. python environment -----------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo  [2/5] Creating virtual environment
  python -m venv .venv || goto :fail
)
echo  [3/5] Installing packages (takes a minute the first time)
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q || goto :fail

if not exist ".env" copy ".env.example" ".env" >nul

echo  [4/5] Running tests
".venv\Scripts\python.exe" -m pytest -q tests || (
  echo  [!] Some tests failed. Open the folder in VS Code and ask Claude to look at it.
)

rem ---- 4. git -----------------------------------------------------------
if not exist ".git" (
  echo  [5/5] Creating git repository
  git init -q -b main
  git add -A
  git commit -q -m "Initial commit: Ultron paper trader" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || (
    echo  [i] Git needs your name and email once. Run these, then run setup again:
    echo      git config --global user.name "Your Name"
    echo      git config --global user.email "you@example.com"
  )
) else (
  echo  [5/5] Git repository already there
)

rem ---- 5. VS Code ---------------------------------------------------------
where code >nul 2>nul && (
  echo.
  echo  Installing the Claude Code and Python extensions in VS Code
  call code --install-extension anthropic.claude-code --force >nul 2>nul
  call code --install-extension ms-python.python --force >nul 2>nul
  echo  Opening VS Code...
  call code "%DEST%"
) || (
  echo.
  echo  [i] VS Code's "code" command isn't on PATH. Open VS Code, File ^> Open Folder,
  echo      and pick %DEST%. It will offer to install the recommended extensions.
)

echo.
echo  Done. Next:
echo   1. Open .env in VS Code and paste your GEMINI_API_KEY and XAI_API_KEY
echo   2. Open the Claude Code panel, sign in with your Claude account
echo   3. Tell it: "Read CLAUDE.md, then run main.py and fix whatever breaks"
echo.
echo  You can delete the copy in Downloads once this all works.
goto :end

:fail
echo.
echo  [!] Something went wrong above. Copy the error and paste it into Claude.

:end
echo.
pause
