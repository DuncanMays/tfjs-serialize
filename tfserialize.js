//'use strict';

let serialjs;
let _local_tf;
let tfserialize;

/**
 * A tf.IOHandler that serializes a model into a string.
 */
class SerializeIOHandler {

	/**
	 * @param {string} model - the serialized model, if loading. Othersize, the
	 * serialized model will go in this.model after saving is complete.
	 */
	constructor(model) {
		this.model = model;
	}

	/**
	 * Serializes the model.
     * @param {tf.IOHandler} modelArtifacts - The weights and topology of the model.
	 * @return {Promise<tf.SaveResult>} - A promise resolving to an object containing error information if applicable. Not often used.
	 */
	async save(modelArtifacts) {
		// In case the library isn't loaded yet, for browser code.
		serialjs = await serialjs;
		this.model = serialjs.serialize(modelArtifacts);
		return {
			modelArtifactsInfo: {
				dateSaved: new Date(),
				// Note that this isn't exactly correct, since we're not using
				// JSON, but rather serialize.js for this.
				modelTopologyType: 'JSON',
				modelTopologyBytes: modelArtifacts.modelTopology == null ?
                  0 : JSON.stringify(modelArtifacts.modelTopology).length,
                weightSpecsBytes: modelArtifacts.weightSpecs == null ?
                  0 : JSON.stringify(modelArtifacts.weightSpecs).length,
                weightDataBytes: modelArtifacts.weightData == null ?
                  0 : modelArtifacts.weightData.byteLength,
			}
		};
	}

	/**
	 * Deserializes the model using this.model.
	 */
	async load() {
		// In case the library isn't loaded yet, for browser code.
		serialjs = await serialjs;

		if (typeof this.model === "undefined") {
			throw new Error("SerializeIOHandler.load() called without providing a model to load.");
		}

		return serialjs.deserialize(this.model);
	}
}
	
/**
 *This function serializes the training information that is needed to train the model
 *@param {tf.LayersModel} - A model that needs its training info serialized
 *@return 
 */
async function getTrainingInfo (model) {
	//model.optimizer.getClassName() returns the class name of the optimizer, so "Adam", or "SGD",
	// it is needed to find the constructor of the optimizer suring deserialization in the worker.
	//model.optimizer.getConfig() returns all the stateful information of the optimizer
	const obj = {optimizerName:await model.optimizer.getClassName(),
		optimizerConfig:await model.optimizer.getConfig(),
		loss:model.loss,
		metrics:model.metrics}

	return await JSON.stringify(obj)
}

/**
 *This function deserializes the training information that is needed to train the model
 *@param {string} - A JSON encoded object containing: a tf.serialisation.ConfigDict, which is config object that described all the stateful information of the otpimizer, the loss, and the metrics
 *@returns {object} - an object the contains: the optimizer, the metrics, and the loss
 */
async function fromTrainingInfo (trainingInfoJSON) {
	const trainingInfo = await JSON.parse(trainingInfoJSON)

	const className = trainingInfo.optimizerName
	const config = trainingInfo.optimizerConfig
		
	//tf.serialization.SerializationMap.getMap().classNameMap returns the class name map that tensorflow uses
	//this object maps class names, eg "Adam" or "SGD", to an array of two objects. The first object is the 
	//constructor of the corresponding class, the second object is a function that parses the classes config
	//object (the thing that holds all its stateful information) into the constructor, and returns an instance
	//of the class. If you define your own optimizer and want to use it with this method, you must add that
	//optimizer to the serialization map using the registerClass function from tensorflow's serialization
	//library (not the serialization library Wes wrote).
	const temp = _local_tf.serialization.SerializationMap.getMap().classNameMap[className]

	const constructor = temp[0]
	const parser = temp[1]

	return {optimizer:parser(constructor, config), loss:trainingInfo.loss, metrics:trainingInfo.metrics}
}



/**
 * Serializes a TensorFlow model, turning it into a string.
 * @param {tf.LayersModel} - A model to serialize
 * @returns {string} - The serialized model.
 */
async function serialize(model) {
	const handler = new SerializeIOHandler();
	const result = await model.save(handler);

	const completeModel = {model:handler.model, trainingInfo: await getTrainingInfo(model)}	

	return await JSON.stringify(completeModel)
}

/**
 * Turns a string back into a TensorFlow model.
 * @param {string} str - The string representing the model
 * @returns {tf.LayersModel} - The original model.
 */
async function deserialize(str) {
	const completeModel = await JSON.parse(str)

	const serializedModel = completeModel.model
	const serializedTrainingInfo = completeModel.trainingInfo

	const handler = new SerializeIOHandler(serializedModel);
	const model =  await _local_tf.loadLayersModel(handler);

	const trainingInfo = await fromTrainingInfo(serializedTrainingInfo)

	model.compile(trainingInfo)

	return model
}

const ENVIRONMENT_IS_NODE = typeof require === 'function' &&
	typeof global === 'object' &&
	typeof global.process === 'object' && typeof global.process.constructor === 'function' &&
	typeof global.process.release === 'object' && global.process.release.name === 'node';
const ENVIRONMENT_IS_WORKER = typeof importScripts === 'function';
const ENVIRONMENT_IS_WEB = typeof window === 'object' && typeof document === 'object' && !ENVIRONMENT_IS_WORKER;

if (ENVIRONMENT_IS_WEB) {
	// In the browser, just script tag this in after tensorflow.
	// Use the tfserialize object this defines.

	// Browser code (expected to be in the same directory as serialize.js)
	async function import_serialize() {
		// Using fetch(./serialize.js) fails if this script isn't located in the root
		// of the web server.
		const path_here = document.currentScript.src;
		const parent_path = path_here.slice(0, path_here.lastIndexOf('/'));
		return eval(await (await fetch(parent_path + "/serialize.js")).text())
	}
	// Leave it as a promise for save and load to await.
	serialjs = import_serialize();
	_local_tf = tf;
	// Put the code into the tfserialize object.
	tfserialize = {
		serialize,
		deserialize
	};
	delete serialize;
	delete deserialize;
} else if (ENVIRONMENT_IS_NODE) {
	// Running in node
	serialjs = require('./serialize');
	_local_tf = require('@tensorflow/tfjs');
	exports.serialize = serialize;
	exports.deserialize = deserialize;
} else {
	// Running in a worker
	serialjs = require('serialize');
	_local_tf = require('tfjs');
	exports.serialize = serialize;
	exports.deserialize = deserialize;
}
