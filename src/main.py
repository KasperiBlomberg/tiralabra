from scripts.train import train
from scripts.evaluate import evaluate, view_predictions


def main():
    """
    Runs a command-line interface for training and evaluating the network.

    The user can choose to:
    1. Train the model by specifying a sample size, number of iterations, and learning rate.
    2. Evaluate the model using a specified sample size.
    3. Exit the program.
    """
    while True:
        print("1. Train the model")
        print("2. Evaluate the model")
        print("3. View predictions")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            sample_size = int(input("Enter the sample size: (max 60000)"))
            if sample_size > 60000:
                print("Sample size cannot be greater than 60000. Setting it to 60000.")
                sample_size = 60000
            iterations = int(input("Enter the number of iterations: "))
            alpha = float(input("Enter the learning rate: "))
            filename = input("Enter the filename to save the model weights: (Press enter to use default) ")
            if not filename:
                filename = "model_weights.npz"
            train(alpha, iterations, sample_size, filename)
        elif choice == "2":
            sample_size = int(input("Enter the sample size: (max 10000)"))
            if sample_size > 10000:
                print("Sample size cannot be greater than 10000. Setting it to 10000.")
                sample_size = 10000
            evaluate(sample_size)
        elif choice == "3":
            view_predictions()
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please enter a valid choice.")


if __name__ == "__main__":
    main()
