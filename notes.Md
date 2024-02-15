All One-Off notes, code or doc specific can be kept here

# venv
python3 -m venv .venv
source .paper_venv/bin/activate

# git
- add remote https://docs.github.com/en/get-started/getting-started-with-git/managing-remote-repositories 


# mlflow
pip install mlflow
mlflow server --host 127.0.0.1 --port 8080

```
import mlflow
mlflow.set_tracking_uri(uri="http://<host>:<port>")
mlflow.set_experiment("experiment_name")
mlflow.start_run("run_name")
```