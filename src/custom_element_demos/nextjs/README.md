# Demo

This demo uses the `<model-explorer-visualizer>` custom element in a Next.js
application. Run the commands below to build the app and start a local dev
server.

```
$ npm install
$ npm run dev
```

Visit `http://localhost:3000`

# Notes

- The custom element works in the client-side script.
- Import the visualizer package in `useEffect` instead of importing it at top
  level.
- This demo uses symlinks to make the `worker.js` file and `static_files`
  directory from `node_modules` available under the `public` directory.

