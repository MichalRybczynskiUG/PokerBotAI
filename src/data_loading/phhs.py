from pathlib import Path
import re

def count_heads_up_NT_from_phhs(file_path : Path) -> int:
    """
    Counts how many Heads-up No Limit Texas Hold'em games are in the phhs file

    Parameters
    ----------
    file_path - path to the phhs file

    Returns
    -------
    number of Heads-up No Limit Texas Hold'em games in the file

    """
    num_head_up_NT = 0

    is_NT = False
    is_HU = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith('variant'):
                _, value_var = line.split('=', 1)
                if value_var.strip().strip("'") == 'NT':
                    is_NT = True

            if line.startswith('antes'):
                _, value_antes = line.split('=', 1)
                value_antes = value_antes.split(',')
                num_of_players = len(value_antes)
                if num_of_players == 2:
                    is_HU = True



            if line.startswith("["):
                if is_NT and is_HU:
                    num_head_up_NT += 1

                is_NT = False
                is_HU = False
    return num_head_up_NT

def load_phhs_file(file_dir) -> list:
        games_list = []
        game_dic = {}
        with open(file_dir, 'r', encoding = 'uth-8') as f:
            for line in f:

                if line.startswith("variant"):
                    _ , value = line.split("=", 1)
                    value = value.strip().strip("'").strip('"')
                    game_dic["variant"] = value

                if line.startswith("antes"):
                    _ , value = line.split("=", 1)
                    value = value.strip().strip("[]")
                    value = value.split(",")
                    value = [float(x.strip()) for x in value]
                    game_dic["antes"] = value

                if line.startswith("actions"):

                    _ , value = line.split("=", 1)
                    value = re.findall(r'["\'](.*?)["\']', value)

                    actions_list = []
                    cards_players = {f"p{i+1}": "" for i in range(1,len(game_dic["anthes"]))} #we save cards that each player has
                    community_cards = "" # cards on the table

                    for action in value:
                        action_dic = {}
                        action = action.split()

                        action_dic["actor"] = action[0]
                        action_dic["action"] = action[1]
                        if action_dic["actor"] == "d": #dealer makes action
                            if action_dic["action"] == "dh": #dealer gives cards to a player
                                action_dic["target"] = action[2] #player who gets cards
                                cards_players[action_dic["target"]] += action[3] #we save cards of the player
                            if action_dic["target"] == "db": #dealer deals community cards
                                community_cards += action[2]
                                action_dic["community_cards"] = community_cards
                        elif bool(re.fullmatch(r"p\d+", action_dic["actor"])): #player makes action
                            action_dic["cards"] = cards_players[action_dic["actor"]]
                            if action_dic["action"] == "cbr":
                                action_dic["cbr_amount"] = action[2]

                        actions_list.append(action_dic)

                    game_dic["actions"] = actions_list

                if line.startswith("["):
                    games_list.append(game_dic)
                    game_dic = {}

        return games_list

