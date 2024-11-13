# sem7

Code for our semester 7 project.

## Running the code

1. Either use Linux or WSL2 on Windows
2. Clone the repository
3. Install `ssh`, `sshpass` and `rsync` on your local machine
4. `cd` into the `ws` directory
5. Create a virtual environment using `python3.12 -m venv .venv`
6. Activate the virtual environment using `source .venv/bin/activate`
7. Install the required Python packages on your local machine using `pip install -r
requirements.txt`
8. Download the PTB-XL dataset from https://physionet.org/content/ptb-xl/1.0.3/
   and extract it to the `ws/data/pbl-xl/` folder
9. Now there should be a `ptbxl_database.csv` file in the `ws/data/ptb-xl/`
   folder
10. Translate `ptbxl_database.csv` using `python translate_ptb.py`, generating a file
    called `ptbxl_database_translated.csv` in the same folder
11. Parse the translated csv file using `python parse_ptb.py`, generating
    `parsed_ptb_*.pkl` files in the same folder
12. Now we're ready for training! However, we will probably want to use the AAU
    AI Lab frontend for this to leverage the powerful GPUs available there
13. Create `.env.local` file in the root directory of the project with the
    following content:
    ```bash
    USER=<AAU-email-address>
    PASS=<AAU-password>
    ```
14. Run `./run.sh` to run [ws/main.py](ws/main.py) which is the main file for the
    project
    1. `run.sh` will sync the [ws](ws) folder to the AAU AI Lab frontend
    2. `run.sh` will SSH into the frontend and run `main.sh`
    3. `main.sh` will run the [ws/main.py](ws/main.py) file inside a pytorch
       container
    4. [ws/results](ws/results) will be synced back to the local machine

## Development

- The repository was configured to be used with VSCode
- Install basedpyright and Ruff VSCode extension for Python linting and import
  organization, respectively.

## Datasets

1. https://data.mendeley.com/datasets/xmbxhscgpr/3
2. https://physionet.org/content/ptb-xl/1.0.3/
3. https://data.mendeley.com/datasets/34rpmsxc4z/2
