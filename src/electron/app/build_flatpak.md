Building for flatpak isn't supported yet. If you are working on adding support via `@electron-forge/maker-flatpak`, you'll likely need the following environment:

```sh
sudo apt install flatpak flatpak-builder elfutils
git config --global safe.bareRepository all
git config --global protocol.file.allow always
```


It will probably take 30 minutes to package.
