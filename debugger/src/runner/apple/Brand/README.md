# Runner brand assets

`debugger-logo-1024.png` is the approved native raster corresponding to
`../../ui/debugger-logo.svg`. `source.sha256` binds that SVG to this raster;
`Tools/generate_project.py` checks the hash before generating a project.

From the repository root, regenerate the Apple asset catalogs with:

```bash
xcrun swift src/runner/apple/Tools/sync_icons.swift src/runner/apple
```

If the SVG changes, re-export its native raster and update the hash together.
Preserve `LICENSE-model-explorer.txt` with these derived assets.
