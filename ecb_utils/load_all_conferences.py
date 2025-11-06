from pathlib import Path
import sys

# Dynamically detect project root
cwd = Path.cwd()
if (cwd / 'ecb_scraper').exists():
    project_root = cwd
elif (cwd.parent / 'ecb_scraper').exists():
    project_root = cwd.parent
else:
    project_root = Path(r"C:\\Users\\Garance Latieule\\Projet_ML\\Analyzing-ECB")

sys.path.insert(0, str(project_root))
print('Project root set to:', project_root)

from ecb_scraper.scraper import load_ecb_conferences  # import only the needed function
print('Import OK: ecb_scraper.scraper.load_ecb_conferences')

# LOAD ALL CONFERENCES FROM 1998 TO 2025 AND SAVE AS CSV

df = load_ecb_conferences(start_year=1998, end_year=2025)
output_path = Path("all_ECB_conferences.csv")
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} ECB conferences to {output_path.resolve()}")

# Time ~ 4 min
