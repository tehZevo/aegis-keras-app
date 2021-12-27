# Aegis Keras Application node

[These](https://keras.io/api/applications/) as protopost-compatible nodes

## Usage
POST `"<base64 encoded image>"` to `/` to receive an [nd-to-json](https://github.com/tehZevo/nd-to-json)-encoded array

## Environment
- `PORT` - the port to listen on
- `APP_NAME` - the Keras app name to use (defaults to "mobilenet_v2")
- `POOLING` - the global pooling to apply at the end of the network (defaults to "max")
- `RESIZE_TO` - resize images to `RESIZE_TO x RESIZE_TO` size before feeding to the network

## TODO
- lots of testing needed
