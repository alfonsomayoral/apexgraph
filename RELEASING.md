# Releasing Apexgraph to PyPI

Apexgraph publishes via **Trusted Publishing** (OIDC) from GitHub Actions — no API
token is ever stored. The release workflow lives in
[`.github/workflows/publish.yml`](.github/workflows/publish.yml) and runs when a
GitHub Release is published. The PyPI distribution name is **`apexgraph`** (the
command and import name remain `apexgraph`).

## One-time setup (per maintainer)

1. **Create a PyPI account** at <https://pypi.org/account/register/> and enable
   two-factor authentication (required).
2. **Add a pending trusted publisher** at
   <https://pypi.org/manage/account/publishing/> with:
   - PyPI Project Name: `apexgraph`
   - Owner: `alfonsomayoral`
   - Repository name: `apexgraph`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

   This claims the project name and links it to this repo's workflow. (Optionally
   do the same on <https://test.pypi.org> first for a dry run.)

## Cutting a release

1. Bump `version` in `pyproject.toml` and move the `CHANGELOG.md` entries from
   `[Unreleased]` under the new version heading. Commit and push.
2. Tag and create the GitHub Release:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   gh release create v0.1.0 --title "v0.1.0" --notes-file <(sed -n '/## \[0.1.0\]/,/## \[/p' CHANGELOG.md)
   ```
   (Or create the release from the GitHub UI.)
3. Publishing the release triggers `publish.yml`: it builds the sdist + wheel,
   runs `twine check`, and uploads to PyPI via OIDC. Watch it with
   `gh run watch`.

## Dry run (no publish)

Run the pipeline manually to validate the build, metadata, and artifact handling
without touching PyPI — the upload step is skipped for non-release events:

```bash
gh workflow run publish.yml
gh run watch
```

## Verify

```bash
uv tool install apexgraph
apexgraph --version
```

A version, once published, cannot be replaced — only yanked. Validate locally
first: `uv build && uvx twine check dist/*`.
