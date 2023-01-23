from colorama import Fore, Style


def warning(message: str):
    print(f"{Fore.YELLOW}{Style.BRIGHT}[WARNING] {Style.RESET_ALL}{Fore.YELLOW}{message}{Style.RESET_ALL}")


def error(message: str):
    print(f"{Fore.RED}{Style.BRIGHT}[ERROR] {Style.RESET_ALL}{Fore.RED}{message}{Style.RESET_ALL}")


def info(message: str):
    print(f"{Fore.BLUE}{Style.BRIGHT}[INFO] {Style.RESET_ALL}{Fore.BLUE}{message}{Style.RESET_ALL}")
