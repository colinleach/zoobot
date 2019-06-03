## Serving Images to Custom Estimators with Tensorflow Serving

I found Tensorflow Serving quite tricky to get my head around:
-  There's a lot of jargon cluttering up what's going on: SignatureDef, SavedModelBuilder, profobuf, gRPC, etc.
- The Serving docs are largely aimed at professional developers
- The examples do not cover the tf.Estimator API, which is how I define my model

I'm writing this blog to help myself understand. I hope you find it useful too.

There are three steps to use your custom estimator to make predictions:
1. Export your estimator with SignatureDefs to define inputs and outputs
2. Start a server that loads your exported estimator and waits for requests
3. Run a client that reads data and makes requests to the server

## 1. Export your Estimator with SignatureDefs

*This is exactly the same as for Python predictions and should be a separate blog post*

Because we're making our estimator available as an API, we need to define what it expects and what it predicts.

Defining the predictions your custom estimator makes is done inside the estimator itself, so let's deal with that first.

### Estimator Output (Predictions)

Within your custom estimator function, you can define the output like so:

    from tensorflow.python.saved_model import signature_constants
    ...

    def custom_estimator_func(features, labels, mode):

        response = ...  # make your predictions as normal

        if mode == tf.estimator.ModeKeys.PREDICT:
            # declare that this estimator will make return tensors like `response`
            export_outputs = {
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(response)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions=response, export_outputs=export_outputs)

PredictOutput is the most general way to define what your estimator will return: it will accept any tensor. This is handy if you don't just want, say, a scalar classification - perhaps you'd also like to return the entire dense layer for inspection. If you know you just want to make a simple prediction, you can use ClassificationOutput or RegressionOutput.


### Estimator Input (Data)


Below, I create input instructions that expects a batch of images, does some preprocessing, then passes the resulting features to my estimator.

    config = {'initial_size': 128, 'channels': 3}  # any metadata you need
    def serving_input_receiver_fn_image(): 
        """
        An input receiver that expects an image array (batch, size, size, channels).
        No args allowed: pass in any config options via closure.
        """

        # placeholder to define the graph. Will be replaced by new data.
        images = tf.placeholder(
            dtype=tf.uint8,  # RGB images have values in the range (0, 255)
            shape=(None, config['initial_size'], config['initial_size'] config['channels']), 
            name='images')
 
        # dict of tensors the estimator will expect. Images as above.
        dict_of_input_tensors = {'examples': images}  

        # apply any preprocessing needed before sending to estimator
        features = input_utils.preprocess_batch(images)

        return tf.estimator.export.ServingInputReceiver(
            features, dict_of_input_tensors)

### Executing the Export

Now we can export our trained estimator using those instructions:


    estimator = tf.estimator.Estimator(
        model_fn=custom_estimator_func,
        ...
    )

    estimator.train(...)

    estimator.export_savedmodel(
        export_dir_base='your/directory/here',
        serving_input_receiver_fn=serving_input_receiver_fn)


## 2. Start a Server to Wait for Requests

Tensorflow Serving can load our saved model and, given requests, use it to make predictions. This is only possible because we carefully defined what our model expects as input, and produces as output.

Don't try to build Tensorflow Serving yourself. It's much easier to use Docker. Use the `tensorflow/serving` image:

    docker pull tensorflow/serving

When you run this image, the following command gets executed (from the [excellent official tutorial]()):

    tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}


    docker run -p 8501:8501 \
    --mount type=bind,source=/path/to/my_model/,target=/models/my_model \
    -e MODEL_NAME=my_model -t tensorflow/serving

https://www.tensorflow.org/tfx/serving/api_rest


## 3. Run a Client to Make Requests

### URL

    POST http://host:port/v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]:predict


`/versions/${MODEL_VERSION}` is optional. If omitted, Tensorflow Serving will use the latest model.

### Body

    {
    // (Optional) Serving signature to use.
    // If unspecifed default serving signature is used.
    "signature_name": <string>,

    // Input Tensors in row format.
    // e.g list of 2 tensors each of [1, 2] shape
    "instances": [ [[1, 2]], [[3, 4]] ]
    }

RGB images are efficiently represented as uint8 integers (0-255). Following JSON convention, Tensorflow Serving will identify uint8 integers when written as above (without decimals or quotes).