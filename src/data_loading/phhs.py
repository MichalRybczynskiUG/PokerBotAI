from pathlib import Path
import re

def count_games_from_phhs(file_path: Path, variant: str = 'NT', num_of_players: int = 2) -> int:
    """
    Counts how many Heads-up No Limit Texas Hold'em games are in the phhs file

    Parameters
    ----------
    file_path - path to the phhs file

    Returns
    -------
    number of Heads-up No Limit Texas Hold'em games in the file

    """
    num_games = 0

    is_variant = False
    is_n_players = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith('variant'):
                _, value_var = line.split('=', 1)
                if value_var.strip().strip("'") == variant:
                    is_variant = True

            if line.startswith('starting_stacks'):
                _, value_starting_stacks = line.split('=', 1)
                value_starting_stacks = value_starting_stacks.split(',')
                num_of_players_game = len(value_starting_stacks)
                if num_of_players_game == num_of_players:
                    is_n_players = True

            if line.startswith("["):
                if is_variant and is_n_players:
                    num_games += 1

                is_variant = False
                is_n_players = False

    return num_games

def load_phhs_file(file_dir) -> tuple[list,list]:

        def load_entire_string(index : int, lines : list[str], end_sign = ']'):
            """
            Concatenates next lines if the list is spread across few lines.
            """
            result = ""
            i = index

            while i < len(lines) and not (lines[i].endswith(end_sign)):
                result += lines[i]
                i += 1

            result += lines[i]
            i += 1

            return result, i

        def string_to_list_floats(string_list: str) -> list[float]:
            result = string_list.strip().strip("[]")
            result = result.split(",")
            result = [float(x.strip()) for x in result]

            return result

        games_list = []
        actions_list = []
        game_dic = {}

        with open(file_dir, 'r', encoding = 'utf-8') as f:
            lines = [line.strip() for line in f]

        i = 0
        while i < len(lines):

            line = lines[i]

            if line.startswith("variant"):
                _ , value = line.split("=", 1)
                value = value.strip().strip("'").strip('"')
                game_dic["variant"] = value
                i += 1

            elif line.startswith("blinds_or_straddles"):
                _ , value = line.split("=", 1)
                value = string_to_list_floats(value)
                game_dic["small_blind"] = value[0]
                game_dic["big_blind"] = value[0]
                i += 1

            elif line.startswith("starting_stacks"):
                _, value = line.split("=", 1)
                value = string_to_list_floats(value)
                game_dic["starting_stacks"] = value
                i += 1

            elif line.startswith("min_bet"):
                _, value = line.split("=", 1)
                value = float(value.strip())
                game_dic["min_bet"] = value
                i+= 1

            elif line.startswith("actions"):

                line, i = load_entire_string(i, lines) #concatenating actions spread across different lines
                _ , value = line.split("=", 1)
                value = re.findall(r'["\'](.*?)["\']', value)

                actions_game = []
                cards_players = {f"p{i+1}": "" for i in range(len(game_dic["starting_stacks"]))} #we save cards that each player has
                community_cards = "" # cards on the table

                for j, action in enumerate(value):
                    action_dic = {}
                    action = action.split()

                    action_dic["action_id"] = j
                    action_dic["actor"] = action[0]
                    action_dic["action"] = action[1]
                    if action_dic["actor"] == "d": #dealer makes action
                        if action_dic["action"] == "dh": #dealer gives cards to a player
                            action_dic["target"] = action[2] #player who gets cards
                            cards_players[action_dic["target"]] += action[3] #we save cards of the player
                        if action_dic["action"] == "db": #dealer deals community cards
                            community_cards += action[2]
                            action_dic["community_cards"] = community_cards
                    elif bool(re.fullmatch(r"p\d+", action_dic["actor"])): #player makes action
                        action_dic["cards"] = cards_players[action_dic["actor"]]
                        if action_dic["action"] == "cbr":
                            action_dic["cbr_amount"] = action[2]

                    actions_game.append(action_dic)

                actions_list.append(actions_game)

            elif line.startswith("[") and game_dic:
                games_list.append(game_dic)
                game_dic = {}
                i += 1

            else:
                i += 1

        return games_list, actions_list
