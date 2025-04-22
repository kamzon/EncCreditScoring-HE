import os
import subprocess

def install_dependencies():
    """
    Install required Python dependencies.
    """
    print("\n=== Installing Required Python Packages ===")
    subprocess.check_call(["pip", "install", "--upgrade", "pip"])
    required_packages = [
        "numpy",
        "pandas",
        "torch",
        "scikit-learn",
        "imbalanced-learn"
    ]
    for package in required_packages:
        subprocess.check_call(["pip", "install", package])
    print("\n=== Dependencies Installed Successfully ===")

def run_train_model():
    """
    Run the train_model.py script.
    """
    print("\n=== Running train_model.py ===")
    os.system("python train_model.py")

def run_compare_results():
    """
    Run the compare_results.py script.
    """
    print("\n=== Running compare_results.py ===")
    os.system("python compare_results.py")

if __name__ == "__main__":
    print("=== Python Setup for Encrypted Logistic Regression ===")
    install_dependencies()

    # Run train_model.py to train the model and generate CSV files
    run_train_model()

    # Run compare_results.py to compare Python vs. C++ results
    run_compare_results()

    print("\n=== Setup Complete ===")