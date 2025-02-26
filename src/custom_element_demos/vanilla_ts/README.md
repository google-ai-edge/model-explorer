# Demo

This demo uses the `<model-explorer-visualizer>` custom element in vanilla TS.
Run the command below to build and launch the demo in a browser tab.

```
$ npm run build_and_deploy
```

It should open the page `http://localhost:8080/dist` automatically in browser.

# Notes

- Import `ai-edge-model-explorer-visualizer` in the main typescript file.
- This demo uses symlinks to make the `worker.js` file and `static_files`
  directory from `node_modules` available in the same directory as `index.html`.
- The `worker.js` file is linked to `dist/my_worker_path/worker.js` instead of
  the default location (`dist/worker.js`). To ensure the custom element can
  load it successfully, the main script must set
  `modelExplorer.workerScriptPath` to this path.

Check out file comments in `src/` for more details.
