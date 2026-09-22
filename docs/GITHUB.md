# Publish the repository

Unzip the archive and enter its `rateless-uq-har` directory. The package contains no GitHub credentials and does not push automatically.

1. Read `docs/REPRODUCIBILITY.md` and confirm which choices match the intended paper revision.
2. Fill in the final authors and publication metadata in `CITATION.cff.example` (rename to `CITATION.cff` after completing it).
3. Add a license chosen by the copyright holders. No license has been assumed.
4. Run `bash scripts/run_tests.sh` and `bash scripts/run_smoke.sh` in a clean virtual environment.
5. Create an empty GitHub repository, then replace `YOUR_ACCOUNT/YOUR_REPOSITORY` below.

```bash
git init -b main
git add .
git commit -m "Add reproducible rateless uncertainty HAR reference implementation"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

`.gitignore` excludes downloaded datasets, training outputs, virtual environments, build products, and caches. The included GitHub Actions workflow runs CPU tests and a synthetic demo. It does not validate hardware results or automatically download benchmark datasets.

If trained artifacts are released later, include exact subject splits, prepared-data hashes, seed/config files, package versions, model hashes, calibration histograms, raw result CSVs, and measurements supporting the published claims. Use repository releases or an appropriate archival service for large artifacts.
