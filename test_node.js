/**
 * @file    main.js
 *          Used for training over the DCP network.
 *          Adapted from the tensorflow.js mnist-core example.
 * @author  Ian Chew
 * @date    Jan 2019
 */
const tf = require('@tensorflow/tfjs');
const tfserialize = require('./tfserialize');

async function go() {
	const model = tf.sequential();

	// Define a simple dense layer with 3 inputs and one output.
	const layer = tf.layers.dense({units:1, inputShape:[3]});
	model.add(layer);

	// Set the weights of the layer to [1,2,3], with bias 0.
	layer.setWeights([tf.tensor([[1],[2],[3]]),tf.tensor([0])]);

	model.compile({optimizer: tf.train.adam(), loss: 'meanSquaredError', metrics: ['accuracy']})

/*
	// Create an IOHandler to use in model.save and model.load.
	const handler = new SerializeIOHandler();

	const results = await model.save(handler);
*/

	const serialized = await tfserialize.serialize(model);

	// The model is saved into handler.model.
	console.log("Serialized model: " + serialized);

	// Reload the model from the serialized format (uses handler.model).
	const loadedModel = await tfserialize.deserialize(serialized);

	console.log("Model reloaded");

	// Input to feed into the model. The model should return 14.
	const input = tf.tensor([[1,2,3]]);

	const originalResult = await model.predict(input);

	const loadedResult = await loadedModel.predict(input);

	console.log ("Result from the original model: " + originalResult.arraySync());

	console.log ("Result from the loaded model: " + loadedResult.arraySync());

	console.log('now testing training ability')

	const data = tf.tensor([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
	const labels = tf.tensor([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])

	await model.fit(data, labels)
	await loadedModel.fit(data, labels)

	console.log('training ability confirmed')
}

go();

