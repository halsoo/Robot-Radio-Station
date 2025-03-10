# Robot Radio Station - AI Playlist Generator

## Project structure
    - robot-radio-station/
      - config/
        - data/
          - <data config>.yaml
        - nn_params/
          - <nn config>.yaml
        - config.yaml
      - rrs/ # Main codes are in here
      - metadata/
        - <split name>/
          - train-segments.csv
          - valid-segments.csv
          - test-segments.csv
      - notebooks/
        - <jupyter>.ipynb
      - notebooks_local/
      - outputs/ (hydra stuffs)
      - wandb/
        - <run id>/ # see train.generate_experiment_name for run_id 
          - checkpoints/
            - <model ckpt>.pt
        - debug/
          - <debug runs>
      - inference.py
      - train.py
      - environment.yml

## Collaboration Guideline
### workflow
1. before starting to add something, make your own branch first.
2. do whatever you want with your Jupyter notebook inside 'notebooks_local/'.
3. move important things you've done inside the notebook to a separate script file (or an existing script file).
4. make a pull request.
5. after merging, remove your branch, or discuss with your collaborator about your branch's future.

### RULEs about Jupyter notebook files
* **DO NOT EDIT any notebooks without discussing with the owner (the person who created the notebook).** 
* **DO NOT EXECUTE notebooks inside 'notebooks/'.**
* **MOVE IMPORTANT THINGS YOU'VE DONE INSIDE THE NOTEBOOK TO A SEPARATE SCRIPT FILE BEFORE MAKING A PULL REQUEST**
* treat notebooks inside 'notebooks/' as a document files. **a notebook file is not a valid format for collaboration.**
* (in most cases) only the owner can edit and commit on the notebooks they created.
* consider 'notebooks_local/' as the main notebook working directory.
  * if you want to execute one of notebooks inside 'notebooks/', make a copy of the notebook inside 'notebooks_local/'.
  * if you want to make a new notebook, make it inside 'notebooks_local/' first.
  * if you want to report something as a notebook file, move the notebook from 'notebooks_local/' to 'notebooks/' and then commit.