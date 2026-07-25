from pathlib import Path
path = Path('scripts/intraday_alert_monitor.py')
lines = path.read_text().splitlines()
fixed = []
for line in lines:
    stripped = line.lstrip()
    if not stripped:
        fixed.append('')
        continue
    lead = len(line) - len(stripped)
    new_lead = 4 * round(lead / 4)
    if new_lead == 0 and lead > 0:
        new_lead = 4
    fixed.append(' ' * new_lead + stripped)
path.write_text('\n'.join(fixed) + '\n')
print('Indentation normalized.')
