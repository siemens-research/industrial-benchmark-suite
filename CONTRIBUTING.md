# Contributing

The primary purpose of this repository is to serve as a reference for upcoming scientific publications, enabling readers to reproduce the experiments and results reported in those publications.

Additional proposals and/or code implementations for new industry-inspired benchmarks from the open-source community are welcome. They will be reviewed and selected for inclusion in the future by the project owners listed in [CODEOWNERS.md](CODEOWNERS.md).

GitHub Actions runs tests and pre-commit checks with Python 3.11, 3.12, and 3.13 on pushes and pull requests. Test and coverage reports are available as workflow artifacts.

## Changelog

Use [Conventional Commits](https://www.conventionalcommits.org/) for commits merged into the default branch, for example `feat: add a benchmark` or `fix: correct a reward`. The changelog includes standard conventional commit subjects; other commits are omitted, except for the initial historical commit. Use a conventional pull request title when squash merging.

Pushing a release tag beginning with `v` followed by a digit, on a commit reachable from the default branch, generates `CHANGELOG.md` and opens or updates a pull request. Review its entries for accuracy and sensitive information before merging. The update is made after tagging and is not included in that release tag. The repository or organization must allow GitHub Actions to create pull requests in Actions settings. Pull requests created with `GITHUB_TOKEN` do not automatically run the usual CI workflows; maintainers must ensure required checks run before merging.

To preview the released changelog locally, install [git-cliff](https://git-cliff.org/) version 2.14.2 and run `git cliff --use-branch-tags --offline --output CHANGELOG.md` from the default branch with its full history and tags available. Do not edit the generated file manually: regeneration will overwrite it.
