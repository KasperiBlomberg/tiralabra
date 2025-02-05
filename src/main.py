from scripts.train import train
from scripts.evaluate import evaluate


def main():
    while True:
        print("1. Train the model")
        print("2. Evaluate the model")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            sample_size = int(input("Enter the sample size: (max 60000)"))
            if sample_size > 60000:
                print("Sample size cannot be greater than 60000. Setting it to 60000.")
                sample_size = 60000
            iterations = int(input("Enter the number of iterations: "))
            alpha = float(input("Enter the learning rate: "))
            train(alpha, iterations, sample_size)
        elif choice == "2":
            sample_size = int(input("Enter the sample size: (max 10000)"))
            if sample_size > 10000:
                print("Sample size cannot be greater than 10000. Setting it to 10000.")
                sample_size = 10000
            evaluate(sample_size)
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please enter a valid choice.")


if __name__ == "__main__":
    main()
