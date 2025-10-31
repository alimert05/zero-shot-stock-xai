from fetcher import Fetcher

def main():
    fetcher = Fetcher()
    fetcher.get_input()
    fetcher.search()
    fetcher.display_results()

if __name__ == "__main__":
    main()