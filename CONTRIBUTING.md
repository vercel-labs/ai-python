# Contributing

We're not accepting community pull requests at this time.
Bug reports and ideas are very welcome: please open an issue or start a discussion instead.

## Releasing

1. Draft the release:

   ```bash
   gh release create vX.Y.Z --generate-notes --draft
   ```

2. Review the draft on GitHub and add migration notes under **Breaking Changes**.
3. Publish the release. This creates the tag, which triggers
   `.github/workflows/publish.yml` and uploads the package to PyPI.

GitHub release notes are the changelog of record; `CHANGELOG.md` only holds
a pointer.
