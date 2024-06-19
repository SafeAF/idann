# Idann
Statistics and analytics with historical weather data. With some ML thrown in for fun.


# Usage
## Use a Python Virtual Environment

Using a virtual environment allows you to manage Python packages independently of the system packages. This is the recommended approach when working on Python projects. Here’s how to set it up:

    Install the necessary tools:

    bash

sudo apt update
sudo apt install python3-venv python3-pip

Create a virtual environment:

bash

python3 -m venv ~/venv

Activate the virtual environment:

bash

source ~/venv/bin/activate

Install Python packages within the virtual environment:

bash

pip install numpy pandas scikit-learn matplotlib

Deactivate the virtual environment when done:

bash

    deactivate

Whenever you work on your project, you'll need to activate the virtual environment with the source ~/venv/bin/activate command.