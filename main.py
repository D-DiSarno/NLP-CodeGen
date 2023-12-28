from train import train
from test import test
from evaluation import eval

while True:
    print("0 - Train")
    print("1 - Test")
    print("3 - Evaluate")
    action = int(input("Select an action: "))

    if action == 0:
        train()
        break
    elif action == 1:
        test()
        break
    elif action == 3:
        eval()
        break
    else:
        print("Please select a valid action.\n")
