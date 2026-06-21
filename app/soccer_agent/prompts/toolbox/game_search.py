from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


def get_extraction_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a helpful assistant that extracts structured match information from natural language text about football matches.\n\n"
            "Extract the following fields and output them EXACTLY in this format (one field per line, no extra words):\n"
            "  league: one of (england_epl, germany_bundesliga, europe_uefa-champions-league, italy_serie-a, france_league-1, spain_laliga) or unknown\n"
            "  season: xxxx-xxxx\n"
            "  date: xxxx-xx-xx\n"
            "  year: xxxx\n"
            "  month: xx\n"
            "  day: xx\n"
            "  time: xx:xx  (kick-off time, NOT a game event timestamp)\n"
            "  score: x - x  (write unknown if the score is not mentioned)\n"
            "  team1: yyy\n"
            "  team2: yyy\n\n"
            "Rules:\n"
            "- 'x' represents a digit; 'yyy' represents a string.\n"
            "- If only one team is mentioned, assign it to team1 and set team2 to unknown.\n"
            "- Use the EXACT team name as it appears in the input — do not normalise or translate it.\n"
            "- For any field that is missing or uncertain, write unknown.\n"
            "- For date: record as xxxx-xx-xx when the full date is available; record year/month/day separately whenever any of those can be determined.\n"
            "- Do NOT output any text other than the structured fields above.\n\n"
            "Use the supplied current date/time context to resolve relative time expressions "
            "(e.g. 'yesterday', 'last week', 'last month', 'last season').\n\n"
            "Also set time_range based on the temporal scope of the query:\n"
            "  'day'   → hôm nay / hôm qua / today / yesterday\n"
            "  'week'  → tuần này / tuần trước / this week / last week\n"
            "  'month' → tháng này / tháng trước / gần đây / recently / this month / last month (default)\n"
            "  'year'  → mùa giải / năm nay / năm ngoái / this season / last season / this year / last year"
        ),
        HumanMessagePromptTemplate.from_template(
            "Current date/time: {time_context}\n\n"
            "{format_instructions}\n\n"
            "Sentence: \"{question}\""
        ),
    ])


def get_match_selection_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a soccer expert that selects the most likely match from a list of candidates.\n\n"
            "A game_id has the format: {league}/{season}/{date}/{home_team}-vs-{away_team}\n"
            "Example: england_epl/2014-2015/2015-02-21/chelsea-vs-burnley\n"
            "(team names are lowercased with spaces replaced by dashes)\n\n"
            "STEP 1 — Exclusion: Remove any candidate where a team name from the original query "
            "cannot be found in that candidate's team names.\n"
            "  Example A: query teams are 'Chelsea' and 'West Ham'; candidate has 'Chelsea FC' and 'Liverpool' "
            "→ exclude (West Ham not found).\n"
            "  Example B: query teams are 'Chelsea' and 'West Ham'; candidate has 'Chelsea FC' and 'West Ham United' "
            "→ keep (both found).\n"
            "  Example C: query team is only 'Chelsea'; candidate has 'Bayern Munich' and 'Real Madrid' "
            "→ exclude (Chelsea not found).\n\n"
            "STEP 2 — Selection: From the remaining candidates:\n"
            "  1. If there is ONE obviously most-probable match → return its game_id exactly in the format: "
            "'The given information seems incomplete, but we found the most probable match in the database "
            "with this game_id: [game_id]. [Short recommendation, e.g. date, score, home/away team.]'\n"
            "  2. If no match is significantly more likely → list all remaining candidates with league, season, "
            "date, score, home_team, away_team, venue, and referee (no game_id), and briefly explain the ambiguity.\n\n"
            "MANDATORY: All text in your response MUST be in ENGLISH."
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: \"{question}\"\n\n"
            "Extracted match info: {info}\n\n"
            "Candidates:\n{candidates}"
        ),
    ])
