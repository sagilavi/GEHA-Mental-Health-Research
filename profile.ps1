function sync-nb {

  python -m jupytext --to notebook `
    .venv\MainMLPipeline\RunningScripts\main.py `
    -o .venv\MainMLPipeline\notebookdebugg\main.ipynb

  python -m jupytext --to notebook `
    .venv\MainMLPipeline\RunningScripts\Classes.py `
    -o .venv\MainMLPipeline\notebookdebugg\Classes.ipynb

  python -m jupytext --to notebook `
    .venv\MainMLPipeline\RunningScripts\Graphics.py `
    -o .venv\MainMLPipeline\notebookdebugg\Graphics.ipynb

  python -m jupytext --to notebook `
    .venv\MainMLPipeline\RunningScripts\Utilities.py `
    -o .venv\MainMLPipeline\notebookdebugg\Utilities.ipynb

  python -m jupytext --to notebook `
    .venv\MainMLPipeline\RunningScripts\Models.py `
    -o .venv\MainMLPipeline\notebookdebugg\Models.ipynb
}
