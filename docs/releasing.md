# Branches and releases

- **`dev`** is the default branch. Feature branches open PRs into `dev`. CI must pass, and
  changes to the fitter must also pass the gate (`python -m bench gate run --tier test`
  against [`baseline/gate.md`](baseline/gate.md)).
- **`main`** is what was last released. It only changes through a release PR from `dev`, or a
  hotfix PR.
- Every package change adds a line under `## Unreleased` in [`CHANGELOG.md`](../CHANGELOG.md).

## Release X.Y.Z

1. On a branch from `dev`, set `version = "X.Y.Z"` in `rust/Cargo.toml` (the only place it is
   written; `pyproject.toml` reads it) and run
   `cargo update --workspace --manifest-path rust/Cargo.toml`. Rename
   `## Unreleased` to `## X.Y.Z (YYYY-MM-DD)` and add a fresh `## Unreleased` above it. Merge
   this into `dev`.
2. Open a PR from `dev` into `main` titled `Release X.Y.Z`. The `release` check confirms that
   `rust/Cargo.lock` agrees, that `CHANGELOG.md` has the section and that tag `vX.Y.Z` does
   not exist yet. Merge it with a merge commit, not a squash, so `dev` and `main` keep the same
   commits.
3. Tag the merge commit and push the tag:
   ```bash
   git switch main && git pull && git tag vX.Y.Z && git push origin vX.Y.Z
   ```
   The Wheels workflow builds four wheels and the sdist, publishes them to PyPI (trusted
   publishing, the `pypi` environment), and creates the GitHub release with the changelog
   section as its notes and the files attached.
4. Bring `dev` level with `main`: `git switch dev && git merge --ff-only main && git push`.

## Hotfix

Branch from `main`, open a PR into `main` with the version bump and changelog entry from step 1,
then follow steps 3–4. Use a plain `git merge main` in step 4 if `dev` has moved on.
