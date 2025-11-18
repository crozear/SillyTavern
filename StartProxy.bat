@echo off
pushd %~dp0
set NODE_ENV=production
node C:\Users\Brendan\Desktop\SillyTavern\st-agent-proxy\v1\chat\completions\server.js
pause
popd
