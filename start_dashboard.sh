#!/bin/bash
# Script to start Buffett Monitor dashboard on startup
cd /home/shalu/buffett-monitor
source venv/bin/activate
nohup venv/bin/streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0 > dashboard.log 2>&1 &
echo "Buffett Monitor dashboard started on port 8501"