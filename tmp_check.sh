#!/bin/bash
cd /home/shalu/buffett-monitor
source venv/bin/activate
LOG_FILE="/home/shalu/buffett-monitor/dashboard_start.log"
echo "$(date): Checking dashboard" >> $LOG_FILE
if ! pgrep -f "streamlit.*dashboard/app.py" > /dev/null; then
    echo "$(date): Starting Buffett Monitor dashboard (no process found)" >> $LOG_FILE
    nohup venv/bin/streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0 > dashboard.log 2>&1 &
else
    if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q "200"; then
        echo "$(date): Restarting Buffett Monitor dashboard (process running but not serving HTTP)" >> $LOG_FILE
        pkill -f "streamlit.*dashboard/app.py"
        nohup venv/bin/streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0 > dashboard.log 2>&1 &
    else
        echo "$(date): Dashboard already running and serving requests" >> $LOG_FILE
    fi
fi