def get_integer(msg: str, min: int=None, max: int=None) -> int:
    while True:
        val = input(msg + " > ")
        try:
            val = int(val)
            if ((min == None) or val >= min) and ((max == None) or val <= max):
                return val
            else:
                print(f"Value is out of range! (min = {min}, max = {max})")
        except KeyboardInterrupt:
            raise
        except:
            print("Invalid entry!")


def confirm(msg: str) -> bool:
    """
    Ask the user for confirmation.
    """
    res = input(msg + " (Y/n) > ")
    if res == 'Y' or res == 'y' or res == 'yes' or res == 'Yes' or res == "":
        return True
    return False

def select_option(options: list):
    for index, option in enumerate(options):
        print(f" {index} - {str(option)}")
    print()
    return options[get_integer("Enter an option " + 
                f"(0-{len(options) - 1})", min = 0, max = len(options) - 1)]