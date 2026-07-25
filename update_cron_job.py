import json
import os
import tempfile
from datetime import datetime, timedelta
HERMES_DIR = os.path.expanduser('~/.hermes')
CRON_DIR = os.path.join(HERMES_DIR, "cron")
JOBS_FILE = os.path.join(CRON_DIR, "jobs.json")

def load_jobs():
    if not os.path.exists(JOBS_FILE):
        return []
    with open(JOBS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("jobs", [])

def save_jobs(jobs):
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(JOBS_FILE), suffix='.tmp', prefix='.jobs_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({"jobs": jobs, "updated_at": datetime.now().isoformat()}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, JOBS_FILE)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

jobs = load_jobs()
now = datetime.now()
for job in jobs:
    if job.get("id") == "f15a2d1b6773":  # buffett_weekly_scanner job ID
        job["last_run_at"] = now.isoformat()
        # Compute next run based on schedule
        schedule = job["schedule"]
        if schedule["kind"] == "interval":
            minutes = schedule["minutes"]
            next_run = now + timedelta(minutes=minutes)
            job["next_run_at"] = next_run.isoformat()
        elif schedule["kind"] == "cron":
            try:
                from croniter import croniter
                base = now
                cron = croniter(schedule["expr"], base)
                next_run = cron.get_next(datetime)
                job["next_run_at"] = next_run.isoformat()
            except Exception:
                # If croniter not available, leave next_run_at as is or set to None?
                # We'll set to None to indicate error
                job["next_run_at"] = None
        else:  # once
            job["next_run_at"] = None
        break

save_jobs(jobs)
print("Updated job f15a2d1b6773")
print(f"Last run at: {job['last_run_at']}")
print(f"Next run at: {job['next_run_at']}")
