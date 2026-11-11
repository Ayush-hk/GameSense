"""Text for the Terms, Privacy, Help and 'About this view' sections.

Every statement here describes what the application actually does or where its data comes from.
"""
from __future__ import annotations

from typing import Any


def _author(cfg: dict[str, Any]) -> str:
    a = (cfg.get("app", {}) or {}).get("author", "").strip()
    return a if a else "the project's author(s)"


def terms(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    contact = (cfg.get("app", {}) or {}).get("contact", "").strip()
    sections = [
        ("1. Acceptance and scope",
         "These terms apply to your use of the GameSense dashboard and its documentation. By using the dashboard you agree to them. "
         "If you do not agree, please do not use it."),
        ("2. About GameSense",
         f"GameSense is a final-year student project on player behaviour analytics, segmentation, churn prediction and game "
         f"preferences. It was designed and developed by {_author(cfg)}. It is provided for study, research and demonstration."),
        ("3. Data sources and licences",
         "The dashboard analyses publicly available datasets. Each remains subject to its own licence and citation requirements:\n\n"
         "- **Open Play** (Ballou et al., Zenodo, DOI 10.5281/zenodo.17536656): released under CC0 1.0 with a re-identification clause. "
         "You must not attempt to identify any participant.\n"
         "- **World of Warcraft Avatar History** (Lee, Chen, Cheng and Lei, ACM MMSys 2011): provided by its authors for research use; "
         "please cite the paper.\n"
         "- **Steam catalogue snapshot** (Steam Insights): refer to the licence of the repository you obtained it from.\n\n"
         "Full source details are in `docs/GAMES_DATA.md` and `docs/DATASET_CARD.md`."),
        ("4. Permitted use",
         "You may use the dashboard and its results for education, research and personal analysis. You agree not to: attempt to identify "
         "or contact any individual in the data, or combine the data with other sources for that purpose; present the results as "
         "coming from a different dataset; or use the dashboard to harass, discriminate against or profile individuals."),
        ("5. Interpretation of results",
         "The dashboard reports patterns in the datasets named above. Results describe the people and periods covered by those datasets "
         "and show associations, not causes. Frustration scores are survey-based indicators of in-game experience; they are not "
         "clinical or diagnostic measures. Nothing in the dashboard is medical, psychological, legal or financial advice. "
         "Please read the **Help & FAQ** page for how each measure is built and how to read it."),
        ("6. Intellectual property",
         "The source code, documentation and design of GameSense belong to its author(s) unless a licence file in the project states "
         "otherwise. The datasets belong to their respective owners."),
        ("7. Availability",
         "The dashboard is provided on an \"as is\" and \"as available\" basis, without warranties of any kind. It runs on your own "
         "computer, and its availability depends on your setup."),
        ("8. Limitation of liability",
         "To the extent permitted by law, the author(s) are not liable for any loss or damage arising from the use of, or reliance "
         "on, the dashboard or its results."),
        ("9. Changes",
         "These terms may be updated as the project develops. The version and date are shown at the top of this page."),
    ]
    if contact:
        sections.append(("10. Contact", f"Questions about these terms: {contact}"))
    return sections


def privacy(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    contact = (cfg.get("app", {}) or {}).get("contact", "").strip()
    sections = [
        ("1. Overview",
         "GameSense runs on your own computer. There are no user accounts, no sign-in and no GameSense server. This policy explains "
         "what the dashboard stores and how the datasets it analyses are protected."),
        ("2. What is stored",
         "- **Your profile settings** (display name, role, organisation, e-mail if you enter one, accent colour and default filters) "
         "are saved in one small file on your computer: `data/user/profile.json`. They are used only to personalise the dashboard.\n"
         "- **Analysis results** produced from the datasets are saved in `reports/` and `data/processed/` on your computer."),
        ("3. What is not collected",
         "The dashboard does not send your profile or the data shown on its pages to any external service. GameSense switches off "
         "Streamlit's anonymous usage statistics in `.streamlit/config.toml`. If you change that file, Streamlit's own settings apply."),
        ("4. Dataset participants",
         "The Open Play data contains pseudonymous participant identifiers. The dashboard never displays them. Results are shown "
         "for groups only, and a group must contain at least 10 participants to appear. The dataset licence prohibits attempts to "
         "re-identify participants, and these terms repeat that rule."),
        ("5. Your choices",
         "You can edit or delete your profile at any time on the **Settings** page (Data & privacy). You can also delete "
         "`data/user/profile.json` yourself. Deleting it returns the dashboard to its default settings."),
        ("6. Children",
         "The datasets contain adults aged 18 and over, and the dashboard is intended for study and research use."),
        ("7. Changes",
         "This policy may be updated as the project develops. The version and date are shown at the top of this page."),
    ]
    if contact:
        sections.append(("8. Contact", f"Privacy questions: {contact}"))
    return sections


FAQ: list[tuple[str, list[tuple[str, str]]]] = [
    ("Getting started", [
        ("What is GameSense?",
         "A dashboard for exploring how people play video games. The **churn project** (pages 1-5) analyses World of Warcraft telemetry "
         "to segment players and predict who becomes inactive. The **Game Lab** (pages 6-10) uses the Open Play study and a Steam "
         "catalogue to show what people play, how that differs by age, which games frustrate players, and which games similar players like."),
        ("How do I rebuild the results?",
         "Run `python -m src.pipeline` for the churn project and `python -m src.games.pipeline` for the Game Lab, then refresh the page."),
        ("How do I change my name, role or colours?",
         "Open **Settings** to change appearance and default filters, or **Account & Profile** to edit your details."),
    ]),
    ("Data and measures", [
        ("Where does the data come from?",
         "Open Play (US and UK adults who shared Xbox, Nintendo and Steam play data and completed a daily diary), the Steam Insights "
         "catalogue, and the WoWAH World of Warcraft polling data. See the Terms page and `docs/GAMES_DATA.md` for sources and licences."),
        ("What does \"frustration\" mean here?",
         "After a day of play, participants rated three statements about the game they played most recently, from 1 (strongly disagree) "
         "to 7 (strongly agree): wishing they could do something else, feeling disappointed with their performance, and finding "
         "interactions with others toxic. Frustration is the average of the three. A score of 3 or more is called \"high\"."),
        ("Why does the age range stop at 43?",
         "The study's participants are adults aged 18 to 43. The dashboard describes that range only."),
        ("What is \"lift\"?",
         "The share of an age band's play time spent in a genre, divided by the share for all participants. 1.0 means the same as "
         "everyone; 1.5 means 50% more than everyone."),
        ("Why is a game or genre missing from a chart?",
         "A group needs at least 10 participants to be shown. Xbox titles are not named in the data (only their genre), and free-to-play "
         "and mobile games are only partly covered by the telemetry."),
    ]),
    ("Methods", [
        ("How are games ranked by frustration?",
         "Each person's entries for a game are averaged first, then games are averaged across people. Scores are adjusted toward the "
         "overall mean when a game has few raters, so small samples do not dominate. The alternative view compares each rating with "
         "the same person's own average, which removes differences in how people use the scale."),
        ("How does the recommender work?",
         "It finds games that are played together: if many people who played 5+ hours of game A also played game B, B is suggested to "
         "someone who likes A. The optional slider lowers the rank of games that players rate as more frustrating than average."),
        ("How should I read the model scores?",
         "ROC-AUC measures how well a model ranks positive cases above negative ones: 0.5 is chance and 1.0 is perfect. Every model is "
         "tested on people or games it did not train on and shown next to simple baselines, so you can see what it adds."),
    ]),
    ("Scope and interpretation", [
        ("Do the results apply to all gamers?",
         "They describe the participants and time periods in the datasets. Open Play includes adults aged 18-43 who chose to share data, "
         "so patterns may differ for other groups. Findings are associations, not causes, and differences between groups are often small."),
        ("Can I rely on a single game's score?",
         "Treat each score as an estimate with a margin: the intervals on the charts show the range. Games with more raters have narrower ranges."),
    ]),
    ("Your data", [
        ("Where is my profile stored, and how do I remove it?",
         "On your computer in `data/user/profile.json`. Use **Settings > Data & privacy** to download or delete it."),
    ]),
]


NOTES = {
    "played": "Hours come from Xbox, Nintendo and Steam telemetry that participants chose to share. Only named titles are ranked, and a game "
              "needs at least 10 players to appear. Xbox titles are not named in the data.",
    "age": "Participants are adults aged 18-43. Lift compares each band with all participants (1.0 = same as everyone). "
           "Age differences are modest and describe different people at different ages, not the same people over time.",
    "frustration": "Frustration averages three diary items (1-7). Scores are adjusted for small samples, intervals show the range, and "
                   "'versus the player's own average' removes differences in how people use the scale. Differences between games "
                   "show association, not cause.",
    "recommender": "Suggestions come from games that are played together. A game counts as 'liked' after 5 hours of play. "
                   "The frustration slider is an optional re-ranking.",
    "models": "Models are evaluated on held-out participants or later releases and compared with simple baselines. "
              "Scores describe how well patterns generalise within these datasets.",
}
