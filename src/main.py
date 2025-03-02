import os
from scripts.train import train
from scripts.evaluate import evaluate, view_predictions


def get_input(prompt, default, type, max_value=None):
    """
    Get user input and return the value entered by the user.

    Args:
        prompt (str): The prompt to display to the user.
        default (str): The default value to return if the user does not enter anything.
        type (type): Whether the input should be an integer or a float.
        max_value (int, optional): The maximum value allowed for the input.

    Returns:
        str: The value entered by the user.
    """
    try:
        value = input(prompt)
        return type(value) if value else default
    except ValueError:
        print("Invalid input. Please enter a valid value.")
        return get_input(prompt, default, type, max_value)


def get_filename_save(prompt, default):
    """
    Get the filename from the user for saving the model weights.

    Args:
        prompt (str): The prompt to display to the user.
        default (str): The default value to return if the user does not enter anything.

    Returns:
        str: The filename entered by the user.
    """
    filename = input(prompt).strip()

    if not filename:
        return default

    if not filename.endswith(".npz"):
        filename += ".npz"

    return filename


def get_filename_load(prompt, default):
    """
    Get the filename from the user for loading the model weights.

    Args:
        prompt (str): The prompt to display to the user.
        default (str): The default value to return if the user does not enter anything.

    Returns:
        str: The filename entered by the user.
    """
    while True:
        filename = input(prompt).strip()

        if not filename:
            return default

        if not filename.endswith(".npz"):
            filename += ".npz"

        if os.path.exists(filename):
            return filename

        else:
            print("File not found. Please enter a valid filename.")
            show_files()


def show_files():
    """
    Show the files in the current directory with a .npz extension.
    """

    print("Files in the current directory:")
    for file in os.listdir():
        if file.endswith(".npz"):
            print(file)


def main():
    """
    Runs a command-line interface for training and evaluating the network.

    The user can choose to:
    1. Train the model by specifying a sample size, number of iterations, and learning rate.
    2. Evaluate the model using a specified sample size.
    3. View predictions made by the model.
    4. Exit the program.
    """
    while True:
        print("1. Train the model")
        print("2. Evaluate the model")
        print("3. View predictions")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            sample_size = get_input(
                "Enter the sample size: (max 60000,  press enter to use default = 60000) ",
                60000,
                int,
                60000,
            )

            iterations = get_input(
                "Enter the number of iterations (Press enter to use default = 100): ",
                100,
                int,
            )

            alpha = get_input(
                "Enter the learning rate: (max 1, press enter to use default = 0.1) ",
                0.1,
                float,
                1,
            )

            filename = get_filename_save(
                "Enter the filename to save the model weights: (Press enter to use default = model_weights.npz) ",
                "model_weights.npz",
            )

            train(alpha, iterations, sample_size, filename)

        elif choice == "2":
            sample_size = get_input(
                "Enter the sample size: (max 10000, press enter to use default = 10000) ",
                10000,
                int,
                10000,
            )

            show_files()

            filename = get_filename_load(
                "Enter the filename to get the model weights from: (Press enter to use default = model_weights.npz) ",
                "model_weights.npz",
            )

            evaluate(sample_size, filename)

        elif choice == "3":
            show_files()

            filename = get_filename_load(
                "Enter the filename to get the model weights from: (Press enter to use default = model_weights.npz) ",
                "model_weights.npz",
            )

            view_predictions(filename=filename)

        elif choice == "4":
            break

        else:
            print("Invalid choice. Please enter a valid choice.")


if __name__ == "__main__":
    main()
