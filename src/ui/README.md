An Angular project for Model Explorer UI.

# Local development

## One-time setup

Install Angular CLI and packages:

```
$ npm install -g @angular/cli
$ npm install
```

## Run dev server

First, start the Model Explorer python server locally for handling API requests.
Check out the [Installation Guide](https://github.com/google-ai-edge/model-explorer/wiki/1.-Installation)
if needed.

```
$ model-explorer --no_open_in_browser
```

Then, start the Angular dev server:

```
$ ng serve
```

Access the app at http://localhost:4200 after the Angular dev server starts.

<br>

> [!IMPORTANT]
> The python server starts at port 8080 by default, which is also the default
> port the Angular dev server sends API requests to. See below if you want to
> use a different port.

To use a different port:

Start the python server at the port:

```
$ model-explorer --port=<port_number> --no_open_in_browser
```

Update the port number in [`src/proxy.conf.json`](https://github.com/google-ai-edge/model-explorer/blob/main/src/ui/src/proxy.conf.json),
then start the Angular dev server.

## Run demo app without python server

This Angular project also has a demo page showing the main
[visualizer component](https://github.com/google-ai-edge/model-explorer/tree/main/src/ui/src/components/visualizer) directly. It renders the pre-converted model graph JSON files from
a TFLite model and a TF model. This allows you to experiment with the visualizer
code without going through the model conversion process.

You only need to run the Angular dev server to see the demo page.


```
$ ng serve
```

Then visit the demo page at http://localhost:4200/demo

# Deployment

Run the following command to build the Angular app and update the corresponding
files in the python package (`src/server/package/src/model_explorer/web_app`).

```
$ npm run deploy
```

<br>

> [!NOTE]
> On Linux host, install `rollup` manually if you see the
> `Cannot find module @rollup/rollup-linux-x64-gnu` error.
>
> `$ npm install @rollup/rollup-linux-x64-gnu`

# Contributions

We are not currently accepting community contributions to Model Explorer UI
(except for links to community-built adapters). Please submit feature requests
or proposals to collaborate via the Issue Tracker.
