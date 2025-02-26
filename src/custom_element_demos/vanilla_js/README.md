# Demo

This demo uses the `<model-explorer-visualizer>` custom element in vanilla JS.
Run the command below to prepare and launch the demo in a browser tab.

```
$ ./build_and_deploy.sh
```

It should open the page `http://localhost:8080/dist` automatically in browser.

# Notes

- Load `https://unpkg.com/ai-edge-model-explorer-visualizer@latest` in a script tag in `index.html`.
- Download `https://unpkg.com/ai-edge-model-explorer-visualizer@latest/dist/worker.js` and place it in the
   same directory as index.html.
- To enable the visualizer to load static files from `unpkg`, set
   `modelExplorer.assetFilesBaseUrl` to `https://unpkg.com/ai-edge-model-explorer-visualizer@latest/dist/static_files`.
   Alternatively, you can download the `static_files` directory and place it in
   the same directory as `index.html` for local serving.

Check out file comments in `src/` for more details.



