"""Game-name normalisation. Diary game names are free text typed by participants, so they are
matched to telemetry titles by a normalised key. The alias table only holds renames that point
to the same product (e.g. 'lol' -> 'league of legends'); anything uncertain is left unmatched."""
from __future__ import annotations

import re
import unicodedata

ALIASES = {
    "lol": "league of legends",
    "wow": "world of warcraft",
    "cs2": "counter strike 2",
    "cs": "counter strike 2",
    "csgo": "counter strike 2",
    "cs go": "counter strike 2",
    "counter strike global offensive": "counter strike 2",
    "counter strike": "counter strike 2",
    "overwatch": "overwatch 2",          # Overwatch 2 replaced Overwatch on the same service
    "ow2": "overwatch 2",
    "ow": "overwatch 2",
    "gta v": "grand theft auto v",
    "gta 5": "grand theft auto v",
    "gta5": "grand theft auto v",
    "gta online": "grand theft auto v",
    "bg3": "baldurs gate 3",
    "baldurs gate iii": "baldurs gate 3",
    "r e p o": "repo",
    "counterstrike 2": "counter strike 2",
    "tft": "teamfight tactics",
    "animal crossing": "animal crossing new horizons",
    "dbd": "dead by daylight",
    "ffxiv": "final fantasy xiv online",
    "final fantasy xiv": "final fantasy xiv online",
    "final fantasy 14": "final fantasy xiv online",
    "osrs": "old school runescape",
    "poe": "path of exile",
    "poe2": "path of exile 2",
    "marvel rivals": "marvel rivals",
    "valorant": "valorant",
    "hsr": "honkai star rail",
    "genshin": "genshin impact",
    "tf2": "team fortress 2",
    "r6": "tom clancys rainbow six siege",
    "rainbow six siege": "tom clancys rainbow six siege",
    "rocket league": "rocket league",
    "apex": "apex legends",
    "mc": "minecraft",
    "pokemon go": "pokemon go",
    "stardew": "stardew valley",
    "skyrim": "elder scrolls v skyrim",
    "elden ring": "elden ring",
    "monster hunter wilds": "monster hunter wilds",
    "mhw": "monster hunter world",
    "destiny": "destiny 2",
    "cod": "call of duty",
    "warzone": "call of duty",
    "call of duty warzone": "call of duty",
}


def normalize(name: object) -> str:
    """Lower-case, strip accents / trademark signs / punctuation, collapse spaces, apply aliases."""
    if name is None or (isinstance(name, float) and name != name):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ").replace("®", "").replace("™", "").replace("©", "")
    s = re.sub(r"['’`]", "", s)                 # baldur's -> baldurs
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"^the ", "", s) if s not in ALIASES.values() else s
    s = re.sub(r"\s+", " ", s)
    return ALIASES.get(s, s)
