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

## Publish on GitHub Pages

The workflow in `.github/workflows/docs.yml` builds the documentation and deploys
it to GitHub Pages. To enable publishing once:

1. Open the repository's [Pages settings](https://github.com/haotian-zhuang/BCRFoundation/settings/pages).
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Open [Actions → Documentation](https://github.com/haotian-zhuang/BCRFoundation/actions/workflows/docs.yml),
   select **Run workflow**, choose `main`, and click **Run workflow**.
4. After both jobs succeed, open
   <https://haotian-zhuang.github.io/BCRFoundation/>.

Subsequent changes to the documentation, figure, package metadata, or workflow on
`main` publish automatically. Pull requests build the documentation without
deploying it. You can also use **Run workflow** to rebuild manually.

The workflow installs only `docs/requirements.txt`. It uploads the generated
HTML site, without research datasets or model weights, and does not execute the
tutorial examples.

See GitHub's [publishing source guide](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
and [custom workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
