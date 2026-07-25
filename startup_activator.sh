#!/bin/bash
# Startup script for Buffett Monitor - starts dashboard and updates agent context
cd /home/shalu/buffett-monitor
source venv/bin/activate

# Start the dashboard if not already running
if ! pgrep -f "streamlit.*dashboard/app.py" > /dev/null; then
    echo "$(date): Starting Buffett Monitor dashboard on startup" >> dashboard_start.log
    nohup venv/bin/streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0 > dashboard.log 2>&1 &
else
    echo "$(date): Dashboard already running on startup" >> dashboard_start.log
fi

# Update agent context by ensuring we have the latest information about our work
# This is handled by the memory system, but we can log that we're active
echo "$(date): Buffett Monitor system activated - Agent ready as stock portfolio manager" >> system_activation.log