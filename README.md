# sem7
Code for our semester 7 project.

## Running the code
1. Clone the repository
2. Create `.env.local` file in the root directory of the project with the following content:
    ```bash
    USER=<AAU-email-address>
    PASS=<AAU-password>
    ```
3. Run `./run.sh` to run [ws/main.py](ws/main.py) which is the main file for the project. 
    1. `run.sh` will sync the [ws](ws) folder to the AAU AI Lab frontend, and then will ssh into the frontend and run `main.sh`.
    2. `main.sh` will run the [ws/main.py](ws/main.py) file inside a pytorch container.
    3. [ws/results](ws/results) will be synced back to the local machine.
