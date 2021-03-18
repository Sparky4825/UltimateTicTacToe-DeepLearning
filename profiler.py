import cProfile

from main import main

if __name__ == '__main__':
    print(cProfile.run("main()"))
