const {signAsync} = require('@electron/osx-sign');

if (process.argv.length != 4) {
  console.error(
    'Error: Invalid args. Please call the script correctly. e.g. npm run darwin-sign -- /path/to/app /path/to/entitlement',
  );
  return;
}

var app_path = process.argv[2];
var entitlement_path = process.argv[3];

signAsync({
  app: app_path,
  optionsForFile: (filePath) => {
    return {
      entitlements: entitlement_path,
    };
  },
})
  .then(function () {
    console.log('signed succeed!');
  })
  .catch(function (err) {
    console.log('signed failed!');
    console.error(err);
  });
