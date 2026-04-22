from pathlib import Path

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

