# Building and publishing the documentation

The documentation uses Sphinx, MyST Markdown, and the Furo theme. It renders the
tutorial code as examples; building the site does not execute training or load
model checkpoints.

## Build locally

From the repository root:

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open `http://localhost:8000` in your browser. If building on an HPC server, forward
port 8000 to your own computer. The generated `docs/_build/` directory is ignored
by Git.

## Publish on Read the Docs

1. Commit and push the documentation source and `.readthedocs.yaml` to GitHub.
2. Sign in to [Read the Docs](https://app.readthedocs.org/) with the GitHub account
   that manages `haotian-zhuang/BCRFoundation`.
3. Select **Add project** and choose that repository. Grant the Read the Docs
   GitHub App access to the repository if requested.
4. Use `main` as the default branch and the root `.readthedocs.yaml` configuration.
   Choose the project slug `bcrfoundation` if it is available.
5. Run the first build and check its published URL. After it succeeds, update the
   README documentation badge/link to the confirmed hosted URL.

Read the Docs installs only `docs/requirements.txt` for this build. It does not
need PyTorch, the analysis environment, research datasets, or model weights.

See the official [project setup guide](https://docs.readthedocs.com/platform/stable/intro/add-project.html)
and [configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html).
