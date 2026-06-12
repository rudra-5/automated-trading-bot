@echo off
REM Single-cycle paper carry run, invoked by Windows Task Scheduler every 15 min.
REM State in state\paper_carry.json carries memory between runs; this just fires
REM one cycle and exits. Output appended to logs\cron_paper.log.
cd /d "C:\Users\Lenovo\Desktop\Programming\automated-trading-bot"
python run_carry_paper.py >> logs\cron_paper.log 2>&1
