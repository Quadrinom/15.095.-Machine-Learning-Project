"""Download and prepare the public AER Project STAR dataset.

Run: python prepare_project_star.py
Outputs: project_star_students.csv, project_star_column_guide.csv
Uses only Python standard library. Does not drop students or impute scores.
"""
import csv
import io
import pathlib
import urllib.request

OUT = pathlib.Path(__file__).resolve().parent
URLS = [
    'https://vincentarelbundock.github.io/Rdatasets/csv/AER/STAR.csv',
    'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/AER/STAR.csv',
]
raw = None
for url in URLS:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode('utf-8-sig')
        print('Downloaded from:', url)
        break
    except Exception as exc:
        print('Failed:', url, type(exc).__name__, str(exc)[:150])
if raw is None:
    local = OUT / 'STAR.csv'
    if local.exists():
        raw = local.read_text(encoding='utf-8-sig')
        print('Using local file:', local)
    else:
        raise SystemExit('Download failed. Download STAR.csv from the first URL, save it next to this script as STAR.csv, and rerun.')

reader = csv.DictReader(io.StringIO(raw))
source_fields = reader.fieldnames or []
assert 'stark' in source_fields and 'schoolidk' in source_fields, 'Unexpected source columns'
index_col = source_fields[0]  # Rdatasets row names (typically 'rownames')
renames = {index_col: 'student_id'}
fields = ['student_id'] + [f for f in source_fields if f != index_col]
extra = ['total_score_k', 'total_score_1', 'total_score_2', 'total_score_3']
records = []
for row in reader:
    clean = {renames.get(f, f): (row.get(f) or '').strip() for f in source_fields}
    for g in ('k', '1', '2', '3'):
        a, b = clean.get('read'+g, ''), clean.get('math'+g, '')
        try:
            clean['total_score_'+g] = str(float(a)+float(b)) if a and b and a.upper()!='NA' and b.upper()!='NA' else ''
        except ValueError:
            clean['total_score_'+g] = ''
    for key,val in clean.items():
        if val.upper() == 'NA':
            clean[key] = ''
    records.append(clean)
if len(records) != 11598:
    raise ValueError(f'Expected 11,598 records but found {len(records)}. Review source before sharing.')
ids = [r['student_id'] for r in records]
assert all(ids) and len(set(ids)) == len(ids), 'IDs missing or repeated'
with (OUT/'project_star_students.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields+extra)
    writer.writeheader(); writer.writerows(records)

descriptions = {
    'student_id':'Dataset row identifier; newly named from the Rdatasets row index, not necessarily the original STAR study ID',
    'gender':'Student gender', 'ethnicity':'Student ethnicity', 'birth':'Student birth quarter',
}
for g,grade in [('k','kindergarten'),('1','first grade'),('2','second grade'),('3','third grade')]:
    descriptions.update({
      'star'+g:f'Observed classroom assignment in {grade} (regular/small/regular-with-aide)',
      'read'+g:f'Reading scaled score in {grade}',
      'math'+g:f'Math scaled score in {grade}',
      'lunch'+g:f'Free-lunch eligibility in {grade}',
      'school'+g:f'School locale category in {grade}; not an identifier',
      'schoolid'+g:f'School ID in {grade}; use for grouped holdouts',
      'system'+g:f'School system ID in {grade}',
      'degree'+g:f'Teacher education level in {grade}',
      'ladder'+g:f'Teacher career ladder status in {grade}',
      'experience'+g:f'Teacher teaching experience in years in {grade}',
      'tethnicity'+g:f'Teacher ethnicity in {grade}',
      'total_score_'+g:f'Reading plus math scaled scores in {grade}; missing when either component missing'
    })
with (OUT/'project_star_column_guide.csv').open('w',newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['column','description','role_note'])
    for field in fields+extra:
        role = ('outcome (not a pre-treatment feature)' if field.startswith(('read','math','total_score')) else
                'treatment (not a feature)' if field.startswith('star') else
                'grouping identifier (not a feature)' if field.startswith(('schoolid','system')) else
                'potential covariate; verify timing before modeling')
        writer.writerow([field, descriptions.get(field, 'See AER STAR documentation for exact definition'), role])
print(f'Wrote {len(records):,} students x {len(fields)+len(extra)} columns')
print('IMPORTANT: School-level grouping must choose a common reference grade and handle missing/moving school IDs.')
