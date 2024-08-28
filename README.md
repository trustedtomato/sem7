# sem7
Code for our semester 7 project.

## Running the code
1. Clone the repository
2. Run `./run.sh` to run [ws/main.py](ws/main.py) which is the main file for the project. 
    1. `run.sh` will sync the [ws](ws) folder to the AAU AI Lab frontend, and then will ssh into the frontend and run `main.sh`.
    2. `main.sh` will run the [ws/main.py](ws/main.py) file inside a pytorch container.
