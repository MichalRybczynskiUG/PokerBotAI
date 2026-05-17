from pathlib import Path
import re

from six import indexbytes


def is_desired_game(index: int = 0, lines: list[str] = None, variant: str = 'NT', num_of_players: int = 2) -> tuple[int, int, bool]:
    """
    Checks first detected game in list of lines of phhs file starting from index.

    Input:
    __________
    index: the index where the search shpuld start
    lines: list with lines of the phhs file


    Returns
    __________
    Tuple with:
     - information where the game started (int)
     - info where the search ended (where the info abut game ended) (int)
     - True if the game is desired or False if it isn't (bool)

    """
    is_variant = False
    is_n_players = False

    index_game_start = index
    i = index

    while i < len(lines):
        line = lines[i]

        if line.startswith('variant'):
            index_game_start = i

            _, value_var = line.split('=', 1)
            if value_var.strip().strip("'") == variant:
                is_variant = True
            i += 1


        elif line.startswith('starting_stacks'):
            _, value_starting_stacks = line.split('=', 1)
            value_starting_stacks = value_starting_stacks.split(',')
            num_of_players_game = len(value_starting_stacks)
            if num_of_players_game == num_of_players:
                is_n_players = True
            i += 1


        elif line.startswith("["):
            if is_variant and is_n_players:
                return index_game_start, i+1, True
            else:
                return index_game_start, i+1, False

        else:
            i += 1

    if is_variant and is_n_players:
        return index_game_start, i, True

    else:
        return index_game_start, i, False



def count_games_from_phhs(file_path: Path, variant: str = 'NT', num_of_players: int = 2) -> int:
    """
    Counts how many Heads-up No Limit Texas Hold'em games are in the phhs file

    Parameters
    ----------
    file_path - path to the phhs file
    variant - variant of the game
    num_of_players - number of desired players

    Returns
    -------
    number of desired games in the file
    """
    num_games = 0
    lines = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]

    i = 0

    while i < len(lines):
        _, i, is_desired = is_desired_game(i, lines, variant = variant, num_of_players = num_of_players)
        if is_desired:
            num_games += 1

    return num_games

def load_phhs_file(file_dir) -> tuple[list,list]:

        def load_entire_string(index : int, lines : list[str], end_sign = ']'):
            """
            Concatenates next lines if the list is stretched over few lines.
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
            """Converts string to the list of floats"""
            result = string_list.strip().strip("[]")
            result = result.split(",")
            result = [float(x.strip()) for x in result]

            return result

        def cards_string_to_list(cards_string: str) -> list[str]:
            """
            Transfroms string of cards to the list of cards

            Input:
            --------
            cards_string - string of cards like "Kd2d2c8s9c"

            Returns:
            --------
            List of strings where each represents single card
            """
            list_cards = []

            card = ""
            for i in range(len(cards_string)):
                card += cards_string[i]
                if ((i+1) % 2 == 0) & (i != 0):
                    list_cards.append(card)
                    card = ""
            return list_cards


        games_list = [] #list of all games
        actions_list = [] #list of all actions in all games
        game_dic = {} #dictionary representing single game

        with open(file_dir, 'r', encoding = 'utf-8') as f:
            lines = [line.strip() for line in f]

        num_of_players = 0

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
                num_of_players = len(value)
                game_dic["small_blind"] = value[0]
                game_dic["big_blind"] = value[1]
                i += 1

            elif line.startswith("starting_stacks"):
                _, value = line.split("=", 1)
                value = string_to_list_floats(value)
                for k in range(len(value)):
                    game_dic[f"starting_stack_{k+1}"] = value[k]
                i += 1

            elif line.startswith("min_bet"):
                _, value = line.split("=", 1)
                value = float(value.strip())
                game_dic["min_bet"] = value
                i+= 1

            elif line.startswith("actions"):

                line, i = load_entire_string(i, lines) #concatenating actions stretched over different lines
                _ , value = line.split("=", 1)
                value = re.findall(r'["\'](.*?)["\']', value)

                actions_game = [] #list of actions in single game
                cards_players = {f"p{i+1}": [] for i in range(num_of_players)} #we save cards that each player has
                community_cards = [] # cards on the table

                for j, action in enumerate(value):
                    action_dic = {}
                    action = action.split()

                    action_dic["action_id"] = j
                    action_dic["actor"] = action[0]
                    action_dic["action"] = action[1]
                    if action_dic["actor"] == "d": #dealer makes action
                        if action_dic["action"] == "dh": #dealer gives cards to a player
                            action_dic["target"] = action[2] #player who gets cards
                            cards_players[action_dic["target"]] += cards_string_to_list(action[3]) #we save cards of the player
                        if action_dic["action"] == "db": #dealer deals community cards
                            community_cards += cards_string_to_list(action[2])
                            for k in range(len(community_cards)):
                                action_dic[f"community_card_{k+1}"] = community_cards[k]
                    elif bool(re.fullmatch(r"p\d+", action_dic["actor"])): #player makes action
                        action_dic["hand_card_1"] = cards_players[action_dic["actor"]][0]
                        action_dic["hand_card_2"] = cards_players[action_dic["actor"]][1]
                        for k in range(len(community_cards)):
                            action_dic[f"community_card_{k + 1}"] = community_cards[k]
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
