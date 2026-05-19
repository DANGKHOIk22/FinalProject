from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
def get_extraction_prompt_template() -> ChatPromptTemplate:
    """Create a prompt template for searching soccer game information based on user queries. It is used in the 'game_search' tool."""

    game_search_prompt_template = ChatPromptTemplate.from_messages(
        [SystemMessage(
            "You are a helpful assistant that extracts structured information from natural language text about football matches. "
        ),
        HumanMessagePromptTemplate.from_template(
        """
        I will give you a sentence about a football match, and you need to extract the following information: league, season, date, time, and two teams. The output must strictly follow the format below:
        league: (england_epl, germany_bundesliga, europe_uefa-champions-league, italy_serie-a, france_league-1, spain_laliga, or unknown)
        season: xxxx-xxxx
        date: xxxx-xx-xx
        year: xxxx
        month: xx
        day: xx
        time: xx:xx (which means when this game kick-off, not the game timestamp of certain event)
        score: x - x (if score is not determined, write 'unknown' for only in this attribute)
        team1: yyy
        team2: yyy

        All above 'x' means a digit!! 'yyy' means a string.

        To be noted, if you can determine only one team, please assign the team to team1 and leave team2 as 'unknown'. If any information is missing or uncertain, write 'unknown'. You have to use the exactly same name of teams as provided in the input text. Do not output any other words.
        For other attributes, if any information is missing or uncertain, write 'unknown'. As for date, you should record in the form of xxxx-xx-xx if you can get the clear date; Meanwhile, as for year, month, day, you need capture as more information point to this game as possible, including year, month, and day, and record them in numbers.

        IMPORTANT — use the current date/time context below to resolve relative time expressions (e.g. "yesterday", "last week", "last month", "last season"):
        Current date/time: {time_context}

        Also set time_range based on the temporal scope of the query:
        - "day"   → hôm nay / hôm qua / today / yesterday
        - "week"  → tuần này / tuần trước / this week / last week
        - "month" → tháng này / tháng trước / gần đây / recently / this month / last month (default)
        - "year"  → mùa giải / năm nay / năm ngoái / this season / last season / this year / last year

        {format_instructions}

        The sentence is: "{question}"
        """
        )
    ])
    return game_search_prompt_template

def get_match_selection_prompt_template() -> ChatPromptTemplate:
    """Create a prompt template for selecting the most relevant match from candidate matches based on user queries. It is used in the 'match_selection' tool."""

    match_selection_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are an soccer expert that selects the most likely match from a list of candidates based on the given information."
        ),
        HumanMessagePromptTemplate.from_template(
        """
        Now we need to identify the game_id for the most probable match from the database based on the question: "{question}".

        A game_id has the format: {{league}}/{{season}}/{{date}}/{{home_team}}-vs-{{away_team}}
        Example: england_epl/2014-2015/2015-02-21/chelsea-vs-burnley
        (team names are lowercased with spaces replaced by dashes)

        Such question has been transformed to the original query information as:
        Extracted Info: {info}

        Here are the candidate matches:
        Candidates:
        {candidates}

        Based on the original query information and the candidate matches above, is there a match that is significantly more likely than the others?

        Firstly, you should exclude those candidates in the following situation:
        1. If **any of the team's name in original query information** is sure not to be in team names from candidates, such candidate cannot be returned anymore, you cannot let such candidate take place in your return answer.
        2. For example, if the original query information contains "Chelsea" and "West Ham", but candidates contains "chelsea FC" and "Liverpool", since such candidate cannot be returned anymore since West Ham is not in candidate information.
        3. For example, if the original query information contains "Chelsea" and "West Ham", but candidates contains "Chelsea FC" and "West Ham United", since such candidate is still possible to be returned since both team names are in candidate information.
        4. For example, if the original query information contains only "Chelsea", but candidates contains "Bayern Munich" and "Real Madrid", since such candidate cannot be returned since Chelsea is not in candidate information.

        After considering the above situation and exclude those candidate having team name unmatched, you should consider the following two situations:

        1. If there are still **obviously** probable answer with all known information correct, please return the game_id of that match EXACTLY in the following format:
        "The given information seems incomplete, but we found the most probable match in the database with this game_id: [The game_id of the **hugely most probable** match]. [Here give some recommendation to complete the information if possible, for example, provide the date or the score of the match, or which team is the home/away team .etc. Use simple and clear words here.]"

        2. If no match is significantly more likely among all the candidates, please return all candidate matches with information of league, season, date, time, score, home_team, away_team, venue and referee (without game_id), and explain that the information provided is too vague. For this situation you only need to summarize with a little bit the games and give a brief reply with some short sentences.

        MANDATORY: All text in your response MUST be in ENGLISH.
        """
        )
    ])
    return match_selection_prompt_template