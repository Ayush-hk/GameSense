"""Visual building blocks for the dashboard (dark theme, header, KPI cards, insight panels).

Everything that comes from data is passed through html.escape before it is placed in markup.
"""
from __future__ import annotations

import html
from typing import Any, Iterable

import streamlit as st

from src.dashboard.profile import ACCENTS, initials, load_profile


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


CSS = """
:root{--gs-accent:__ACCENT__;--gs-card:#121a35;--gs-card2:#17214a;--gs-border:#25305a;--gs-text:#e6e9f5;--gs-muted:#8b93b5;--gs-good:#22c55e;--gs-warn:#f59e0b;--gs-pink:#ec4899}
.stApp{background:radial-gradient(1200px 500px at 85% -10%,#1b2358 0%,rgba(11,16,32,0) 60%),#0b1020}
.block-container{padding-top:2.2rem;max-width:1320px}
[data-testid="stSidebar"]{background:#0d1330;border-right:1px solid var(--gs-border)}
[data-testid="stSidebarNav"]::before{content:"GameSense";display:block;font-size:24px;font-weight:700;padding:6px 20px 10px;background:linear-gradient(90deg,#fff,var(--gs-accent));-webkit-background-clip:text;background-clip:text;color:transparent}
[data-testid="stSidebarNav"] a{border-radius:10px;margin:1px 8px}
[data-testid="stSidebarNav"] a[aria-current="page"]{background:var(--gs-accent);color:#fff}
.gs-hero{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin:0 0 18px}
.gs-badge{display:inline-block;background:rgba(124,92,255,.18);color:#c9bdff;border:1px solid rgba(124,92,255,.45);border-radius:6px;padding:2px 9px;font-size:11px;letter-spacing:.08em;font-weight:600;margin-bottom:8px}
.gs-hero h1{font-size:34px;line-height:1.15;margin:0;padding:0;font-weight:700;color:#fff}
.gs-sub{color:var(--gs-muted);font-size:15px;margin-top:6px;max-width:760px}
.gs-chips{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}
.gs-chip{background:var(--gs-card);border:1px solid var(--gs-border);border-radius:12px;padding:9px 14px;min-width:150px}
.gs-chip .l{font-size:11px;color:var(--gs-muted)}.gs-chip .v{font-size:13px;color:var(--gs-text);font-weight:600}
.gs-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin:6px 0 18px}
.gs-kpi{display:flex;gap:14px;align-items:center;background:var(--gs-card);border:1px solid var(--gs-border);border-radius:14px;padding:16px 18px}
.gs-ico{width:46px;height:46px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;flex:none;background:linear-gradient(135deg,var(--gs-accent),#4b3bd6)}
.gs-ico.green{background:linear-gradient(135deg,#16a34a,#0f766e)}.gs-ico.pink{background:linear-gradient(135deg,#ec4899,#be185d)}.gs-ico.blue{background:linear-gradient(135deg,#3b82f6,#1d4ed8)}.gs-ico.amber{background:linear-gradient(135deg,#f59e0b,#b45309)}
.gs-kl{font-size:12.5px;color:var(--gs-muted)}.gs-kv{font-size:24px;font-weight:700;color:#fff;line-height:1.2}.gs-ks{font-size:12px;color:#9fb0ff}
.gs-panel{background:var(--gs-card);border:1px solid var(--gs-border);border-radius:14px;padding:16px 18px;margin-bottom:14px}
.gs-panel h3{font-size:16px;margin:0 0 12px;color:#fff;font-weight:650}
.gs-row{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:10px 0;border-top:1px solid var(--gs-border)}
.gs-row:first-of-type{border-top:none}
.gs-row .t{font-weight:600;color:#fff;font-size:14px}.gs-row .d{color:var(--gs-muted);font-size:12.5px}
.gs-pill{background:rgba(236,72,153,.15);color:#f9a8d4;border-radius:8px;padding:3px 9px;font-size:12px;font-weight:600;white-space:nowrap}
.gs-pill.g{background:rgba(34,197,94,.15);color:#86efac}
.gs-note{background:rgba(124,92,255,.10);border:1px solid rgba(124,92,255,.35);border-radius:12px;padding:12px 14px;font-size:13px;color:#d7d0ff;margin:10px 0}
.gs-bars .r{display:grid;grid-template-columns:22px 1fr auto;column-gap:10px;row-gap:5px;align-items:center;margin:11px 0;font-size:13.5px;color:var(--gs-text)}
.gs-bars .n{color:var(--gs-muted)}.gs-bars .nm{white-space:normal;line-height:1.25}
.gs-bars .b{grid-column:2 / 4;height:8px;background:#1d2850;border-radius:6px;overflow:hidden}.gs-bars .b i{display:block;height:100%;background:linear-gradient(90deg,var(--gs-accent),var(--gs-pink));border-radius:6px}
.gs-bars .v{font-weight:600;color:#fff;text-align:right}
.gs-user{display:flex;gap:10px;align-items:center;background:var(--gs-card);border:1px solid var(--gs-border);border-radius:12px;padding:10px 12px;margin:10px 4px}
.gs-av{width:36px;height:36px;border-radius:50%;background:var(--gs-accent);display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;font-size:13px;flex:none}
.gs-user .n{font-size:13.5px;font-weight:600;color:#fff}.gs-user .r{font-size:11.5px;color:var(--gs-muted)}
.gs-legal h2{font-size:18px;margin-top:22px;color:#fff}.gs-legal p,.gs-legal li{color:#c9cfe8;font-size:14.5px;line-height:1.6}
[data-baseweb="tab-list"]{gap:6px;background:var(--gs-card);padding:4px;border-radius:12px;border:1px solid var(--gs-border)}
button[data-baseweb="tab"]{border-radius:9px;padding:8px 16px}
button[data-baseweb="tab"][aria-selected="true"]{background:var(--gs-accent);color:#fff}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none}
[data-testid="stExpander"]{background:var(--gs-card);border:1px solid var(--gs-border);border-radius:12px}

.stApp{background:radial-gradient(900px 420px at 88% -8%,rgba(99,70,255,.34) 0%,rgba(11,16,32,0) 62%),radial-gradient(700px 400px at -5% 105%,rgba(236,72,153,.10) 0%,rgba(11,16,32,0) 60%),#0a0e1f}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0e1536 0%,#0a0f27 100%)}
[data-testid="stSidebarNav"]::before{content:none}
[data-testid="stSidebarNav"] a{padding:6px 12px;font-size:14px;color:#c9cfe8}
[data-testid="stSidebarNav"] a:hover{background:rgba(124,92,255,.14)}
[data-testid="stSidebarNav"] a[aria-current="page"]{background:linear-gradient(90deg,var(--gs-accent),#5b3df5);box-shadow:0 4px 18px rgba(91,61,245,.35);color:#fff;font-weight:600}
[data-testid="stSidebarNav"] li span[class*="Header"],[data-testid="stNavSectionHeader"],[data-testid="stSidebarNav"] [data-testid*="Header"]{color:#8b93b5;letter-spacing:.09em;font-size:11.5px;text-transform:uppercase;font-weight:600}
.gs-hero{position:relative;padding:6px 0 4px;min-height:96px}
.gs-hero{display:flex;align-items:center;gap:18px}
.gs-herotext{flex:1 1 auto;min-width:0}
.gs-art{flex:0 0 190px;height:108px;opacity:.95;filter:drop-shadow(0 0 16px rgba(124,92,255,.5));background:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='130' viewBox='0 0 220 130'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='%237c5cff'/><stop offset='1' stop-color='%23ec4899'/></linearGradient></defs><path d='M60 30h100c22 0 38 30 44 62 3 16-12 26-26 16l-22-18H74l-22 18c-14 10-29 0-26-16 6-32 22-62 44-62z' fill='none' stroke='url(%23g)' stroke-width='5' stroke-linejoin='round'/><path d='M70 58v22M59 69h22' stroke='url(%23g)' stroke-width='6' stroke-linecap='round'/><circle cx='150' cy='60' r='6' fill='url(%23g)'/><circle cx='168' cy='74' r='6' fill='url(%23g)'/></svg>") no-repeat center/contain}
.gs-chipbox{display:flex;background:linear-gradient(135deg,rgba(23,33,74,.85),rgba(14,21,54,.85));border:1px solid var(--gs-border);border-radius:14px;padding:12px 6px;box-shadow:0 8px 30px rgba(0,0,0,.25);z-index:1}
.gs-chipbox .gs-chip{background:none;border:none;border-radius:0;padding:2px 18px;min-width:0}
.gs-chipbox .gs-chip+.gs-chip{border-left:1px solid var(--gs-border)}
.gs-kpi{background:linear-gradient(135deg,rgba(124,92,255,.16),var(--gs-card) 62%);border-color:rgba(124,92,255,.38);box-shadow:0 6px 24px rgba(0,0,0,.22)}
.gs-kpi.green{background:linear-gradient(135deg,rgba(22,163,74,.20),var(--gs-card) 62%);border-color:rgba(34,197,94,.40)}
.gs-kpi.pink{background:linear-gradient(135deg,rgba(236,72,153,.20),var(--gs-card) 62%);border-color:rgba(236,72,153,.40)}
.gs-kpi.blue{background:linear-gradient(135deg,rgba(59,130,246,.20),var(--gs-card) 62%);border-color:rgba(59,130,246,.40)}
.gs-kpi.amber{background:linear-gradient(135deg,rgba(245,158,11,.18),var(--gs-card) 62%);border-color:rgba(245,158,11,.40)}
.gs-panel{background:linear-gradient(160deg,rgba(23,33,74,.75),rgba(15,22,52,.92));box-shadow:0 6px 24px rgba(0,0,0,.22)}
.gs-panel.key{border-color:rgba(124,92,255,.5);background:linear-gradient(160deg,rgba(60,45,140,.45),rgba(15,22,52,.92))}
[data-baseweb="tab-list"]{width:100%;padding:4px}
button[data-baseweb="tab"]{flex:1;justify-content:center}
button[data-baseweb="tab"][aria-selected="true"]{background:linear-gradient(90deg,var(--gs-accent),#5b3df5);box-shadow:0 4px 16px rgba(91,61,245,.35)}
.gs-dataset{display:flex;gap:10px;align-items:center;background:rgba(23,33,74,.6);border:1px solid var(--gs-border);border-radius:12px;padding:10px 12px;margin:10px 4px}
.gs-dataset .n{font-size:12.5px;color:#fff;font-weight:600}.gs-dataset .r{font-size:11px;color:var(--gs-muted)}
.gs-dot{width:8px;height:8px;border-radius:50%;background:var(--gs-good);box-shadow:0 0 8px var(--gs-good);flex:none}.gs-dot.off{background:var(--gs-warn);box-shadow:0 0 8px var(--gs-warn)}
.gs-kpis{grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
.gs-kpi{gap:12px;padding:14px 16px;min-width:0}
.gs-kpi>div:last-child{min-width:0}
.gs-ico{width:40px;height:40px;font-size:19px}
.gs-kv{font-size:22px;white-space:nowrap}
.gs-kl,.gs-ks{overflow-wrap:anywhere}
@media (max-width:1100px){.gs-art{display:none}}
.gs-sub{max-width:640px}
.gs-chipbox{flex:none}
.gs-hero h1{overflow-wrap:anywhere}
"""


def apply_theme(accent_name: str | None = None) -> None:
    accent = ACCENTS.get(accent_name or "", ACCENTS["Violet"])
    st.markdown(f"<style>{CSS.replace('__ACCENT__', accent)}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str = "", badge: str = "", chips: Iterable[tuple[str, str]] = ()) -> None:
    b = f'<div class="gs-badge">{esc(badge.upper())}</div>' if badge else ""
    s = f'<div class="gs-sub">{esc(subtitle)}</div>' if subtitle else ""
    c = "".join(f'<div class="gs-chip"><div class="l">{esc(k)}</div><div class="v">{esc(v)}</div></div>' for k, v in chips)
    box = f'<div class="gs-chipbox">{c}</div>' if c else ""
    st.markdown(f'<div class="gs-hero"><div class="gs-herotext">{b}<h1>{esc(title)}</h1>{s}</div><div class="gs-art"></div>{box}</div>', unsafe_allow_html=True)


def kpis(items: Iterable[dict[str, str]]) -> None:
    """items: dict(label, value, sub='', icon='', tone='')"""
    parts = []
    for i in items:
        tone = esc(i.get("tone", ""))
        small = ' style="font-size:19px"' if len(str(i["value"])) > 12 else ""
        parts.append(
            f'<div class="gs-kpi {tone}"><div class="gs-ico {tone}">{esc(i.get("icon", ""))}</div>'
            f'<div><div class="gs-kl">{esc(i["label"])}</div><div class="gs-kv"{small}>{esc(i["value"])}</div>'
            f'<div class="gs-ks">{esc(i.get("sub", ""))}</div></div></div>'
        )
    cards = "".join(parts)
    st.markdown(f'<div class="gs-kpis">{cards}</div>', unsafe_allow_html=True)


def panel(title: str, rows: Iterable[dict[str, str]]) -> None:
    """rows: dict(title, text='', pill='')"""
    body = "".join(
        f'<div class="gs-row"><div><div class="t">{esc(r["title"])}</div><div class="d">{esc(r.get("text", ""))}</div></div>'
        + (f'<div class="gs-pill">{esc(r["pill"])}</div>' if r.get("pill") else "")
        + "</div>"
        for r in rows
    )
    st.markdown(f'<div class="gs-panel"><h3>{esc(title)}</h3>{body}</div>', unsafe_allow_html=True)


def text_panel(title: str, lines: Iterable[str], key: bool = False) -> None:
    body = "".join(f'<div class="gs-row"><div class="d" style="color:#c9cfe8">{esc(x)}</div></div>' for x in lines)
    st.markdown(f'<div class="gs-panel{" key" if key else ""}"><h3>{esc(title)}</h3>{body}</div>', unsafe_allow_html=True)


def bar_list(title: str, rows: list[tuple[str, float]], max_value: float | None = None, fmt: str = "{:.2f}") -> None:
    if not rows:
        return
    top = max_value or max(v for _, v in rows) or 1.0
    body = "".join(
        f'<div class="r"><div class="n">{i}</div><div class="nm" title="{esc(n)}">{esc(n)}</div><div class="v">{esc(fmt.format(v))}</div>'
        f'<div class="b"><i style="width:{min(100.0, max(2.0, 100 * v / top)):.0f}%"></i></div></div>'
        for i, (n, v) in enumerate(rows, 1)
    )
    st.markdown(f'<div class="gs-panel gs-bars"><h3>{esc(title)}</h3>{body}</div>', unsafe_allow_html=True)


def layout(ratio: float = 1.25):
    """Main column (KPIs, tabs, charts) and a right-hand insight column, as on every page."""
    return st.columns([3, ratio])


def note(text: str) -> None:
    st.markdown(f'<div class="gs-note">{esc(text)}</div>', unsafe_allow_html=True)


def chart(fig) -> None:
    """Plotly figure in the dark theme."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=30, b=10), font=dict(color="#c9cfe8"))
    st.plotly_chart(fig)


def sidebar_card(profile: dict[str, Any]) -> None:
    name = profile.get("display_name", "Guest")
    role = profile.get("role", "")
    st.sidebar.markdown(
        f'<div class="gs-user"><div class="gs-av">{esc(initials(name))}</div>'
        f'<div><div class="n">{esc(name)}</div><div class="r">{esc(role)} - local profile</div></div></div>',
        unsafe_allow_html=True,
    )


def dataset_card() -> None:
    """Sidebar status card: whether the result files the pages read are present."""
    ready = False
    try:
        from src.config import get_paths, load_config
        paths = get_paths(load_config())
        ready = paths.dataset_profile.is_file()
        games = (paths.root / "reports" / "games" / "summary.json").is_file() if hasattr(paths, "root") else True
        ready = ready and games
    except Exception:  # missing config or results: show the neutral state
        ready = False
    dot = "" if ready else " off"
    txt = "Ready" if ready else "Results not found - see Help"
    st.sidebar.markdown(
        f'<div class="gs-dataset"><div class="gs-dot{dot}"></div><div><div class="n">Dataset - real-world data</div><div class="r">{esc(txt)}</div></div></div>',
        unsafe_allow_html=True,
    )


def about_view(profile: dict[str, Any], text: str) -> None:
    """Short 'About this view' note, hidden when the user switched notes off in Settings."""
    if profile.get("show_notes", True):
        with st.expander("About this view"):
            st.markdown(text)


def apply_profile() -> dict[str, Any]:
    p = load_profile()
    apply_theme(p.get("accent"))
    sidebar_card(p)
    dataset_card()
    return p
