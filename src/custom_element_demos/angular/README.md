# Demo

This demo uses the `<model-explorer-visualizer>` custom element in an Angular
application. Run the commands below to build the app and start a local dev
server.

```
$ npm install
$ npm run start
```

It should open the page `http://localhost:4200` automatically in browser.

# Notes

- This demo has a left-side panel to showcase how various visualizer APIs work.
  See comments in `src/app/app.component.ts` for more details.
- This demo uses symlinks to make the `worker.js` file and `static_files`
  directory from `node_modules` available under the `public` directory.
