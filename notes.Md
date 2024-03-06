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

# notes
1. quantiles of evaluation are based off the simulated event times, not the true survival curve. This allows comparison at the .1, .25, .5, .75, .9 quantiles represented by the the event times. The idea is that this will reduce sparsity due to the random sample event times.