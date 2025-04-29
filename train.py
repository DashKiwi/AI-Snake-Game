import os
from agent import VectorizedAgent

def train():
    os.system('cls||clear')
    while True:
        try:
            n_envs = int(input("Snakes to run in the background: "))
            break
        except:
            print("Please enter a valid Intiger")
    while True:
        try:
            visual_choice = input("Do you want a visual interface of one snake? (Y/N) ").upper()
            if visual_choice == "Y":
                visual_choice = True
            elif visual_choice == "N":
                visual_choice = False
            else:
                raise
            break
        except:
            print("Please choose Yes (Y) or No (N)")
    while True:
        try:
            plot_choice = input("Do you want to see plotting of the scores? (Y/N) ").upper()
            if plot_choice == "Y":
                plot_choice = True
            elif plot_choice == "N":
                plot_choice = False
            else:
                raise
            break
        except:
            print("Please choose Yes (Y) or No (N)")
    os.system('cls||clear')
    agent = VectorizedAgent(num_envs=n_envs)
    agent.train(visual=visual_choice, plotting=plot_choice)

if __name__ == '__main__':
    train()

