# Shared Python contracts

`model-debugger-contracts` provides
`model_debugger_contracts.errors.InputRejected` and the model identity functions
`describe_model` / `verify_model`. It imports only the Python standard library
and is shared by model execution and Server registration.

From the repository root:

```sh
python3 -m pip install -e src/contracts/python
PYTHONPATH=src/contracts/python/package python3 -m unittest discover \
  -s src/contracts/python/tests -v
```

Model identity hashes the contents of the model's runtime input files, including
tokenizer data and templates. Hugging Face snapshot symlinks are resolved to
their contents. Verification rejects modified files rather than reusing a stale
registration. Tests use synthetic bytes and do not perform inference.

## Schemas

`model_debugger_contracts/schemas/` holds the JSON Schemas for `CaptureJob`,
`TapManifest` and `capture_index` v2.
`model_debugger_contracts.schema.validate(name, document)` implements the subset
of JSON Schema those files use and raises `SchemaError` with the member path.
The Server validates the jobs it sends and the manifests and indexes it reads;
the PyTorch Runner validates the index it writes;
`src/contracts/fixtures/capture-job.json` is shared with the Apple Runner tests.

`runner-wire.schema.json` describes every message of the Server ↔ Runner
WebSocket protocol (v4 and v5) under `$defs` as `<sender>.<type>`;
`validate_wire(sender, message)` checks one message.
`src/contracts/fixtures/wire/<sender>/` holds one example per message (and per
variant, such as `upload_chunk.v4` and `upload_chunk.v5-header`). The contracts
tests require a fixture for every message in the schema, the Server tests
validate everything its transport sends and is given, and the Apple Runner tests
decode the Server fixtures and compare the members of their replies with the
Runner fixtures. Change a message in the schema and its fixture first; both test
suites then say what else has to move.
