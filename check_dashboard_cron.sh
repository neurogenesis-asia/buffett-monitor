#!/bin/bash
cd /home/shalu/buffett-monitor
source venv/bin/activate
if ! pgrep -f "streamlit.*dashboard/app.py" > /dev/null; then
    echo "$(date): Starting Buffett Monitor dashboard (no process found)" >> dashboard_start.log
    nohup venv/bin/streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0 > dashboard.log 2>&1 &
else
    http_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501)
    if [ "$http_code" -ne 200 ]; then
        echo "$(date): Restarting Buffett Monitor dashboard (process running but not serving HTTP, code: $http_code)" >> dashboard_start.log
        pkill -f "streamlit.*dashboard/app.py"
        nohup venv/bin/streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0 > dashboard.log 2>&1 &
    else
        echo "$(date): Dashboard already running and serving requests" >> dashboard_start.log
    fi
fi