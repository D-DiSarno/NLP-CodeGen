from train import train
from test import test
from evaluation import eval
from demo import demo_presentazione
while True:
    print("0 - Train")
    print("1 - Test")
    print("2 - Evaluate")
    print("3 - Demo")
    action = int(input("Select an action: "))

    if action == 0:
        train()
        break
    elif action == 1:
        test()
        break
    elif action == 2:
        eval()
        break
    elif action == 3:
        demo_presentazione()
        break
    else:
        print("Please select a valid action.\n")
