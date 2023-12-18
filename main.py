from train import train
from test import test

while True:
    print("0 - Train")
    print("1 - Test")
    action = int(input("Select an action: "))

    if action == 0:
        train()
        break
    elif action == 1:
        test()
        break
    else:
        print("Please select a valid action.\n")
