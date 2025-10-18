# Run it locally

Run the following commands under the top `my_ndp` directory:

```shell
# At <repo_root>/src/example_node_data_providers/my_ndp:
#
# Setup python venv.
$ python3 -m venv venv
$ source venv/bin/activate

# Run Model Explorer with the "my_ndp" extension.
#
# Note the "." at the end. This will also install ai-edge-model-explorer.
$ pip install -e .
$ model-explorer coco-ssd.json --extension=my_ndp
```

In the Model Explorer UI, click the "+ Add per-node data" button in the toolbar.
This will open a menu where you should select "My test node data provider" to
launch the configuration dialog. The dialog allows you to try out different
config editors; however, for this extension, only the "Start color" and
"End color" fields will be used to specify the node data's gradient colors.
Set these two colors to distinct values, and then click Run to view the results.
