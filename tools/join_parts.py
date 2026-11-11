"""Join split data files back together:  python tools/join_parts.py

The big Open Play file steam.csv.gz was sent in parts (steam.csv.gz.part1, .part2). This joins them and checks the SHA-256
against the value in docs/GAMES_DATA.md.
"""
import hashlib
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "data" / "raw_games" / "open_play" / "steam.csv.gz"
EXPECTED = "1e54f473aa2c7863617d0348f9eec44c2c0814d201b6add9c7f1a4e1abecd8ce"

parts = sorted(TARGET.parent.glob("steam.csv.gz.part*"))
if not parts:
    sys.exit(f"No parts found next to {TARGET}. Extract all the data zips into the GameSense folder first.")
h = hashlib.sha256()
with open(TARGET, "wb") as out:
    for p in parts:
        data = p.read_bytes()
        h.update(data)
        out.write(data)
if h.hexdigest() != EXPECTED:
    TARGET.unlink()
    sys.exit("Checksum does not match; a part is missing or damaged. Extract the zips again.")
for p in parts:
    p.unlink()
print("steam.csv.gz restored and verified.")
