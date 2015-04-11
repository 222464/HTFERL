#include "HTFERL.h"

#include <iostream>
#include <algorithm>

using namespace htferl;

struct Uint2 {
	unsigned int _x, _y;
};

struct Float2 {
	float _x, _y;
};

struct Float4 {
	float _x, _y, _z, _w;
};

struct Int2 {
	int _x, _y;
};

void HTFERL::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<htfe::LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, int actionQRadius, float minInitWeight, float maxInitWeight, std::mt19937 &generator) {
	std::uniform_int_distribution<int> seedDist(0, 99999);

	_htfe.createRandom(cs, program, inputWidth, inputHeight, layerDescs, minInitWeight, maxInitWeight);

	_inputTypes = inputTypes;

	std::uniform_real_distribution<float> weightDist(minInitWeight, maxInitWeight);
	std::uniform_real_distribution<float> actionDist(0.0f, 1.0f);

	_prevValue = 0.0f;

	_actionQRadius = actionQRadius;

	_input.clear();
	_input.resize(inputWidth * inputHeight);

	int numReconstructionWeightsFirst = std::pow(_actionQRadius * 2 + 1, 2);

	// Initialize action portions randomly
	for (int i = 0; i < _input.size(); i++)
		if (_inputTypes[i] == _action) {
			float value = actionDist(generator);

			_input[i] = value;

			ActionNode actionNode;

			actionNode._index = i;

			actionNode._connections.resize(numReconstructionWeightsFirst);

			for (int j = 0; j < numReconstructionWeightsFirst; j++)
				actionNode._connections[j]._weight = weightDist(generator);

			_actionNodes.push_back(actionNode);
		}
		else if (_inputTypes[i] == _q) {
			QNode qNode;

			qNode._index = i;

			qNode._connections.resize(numReconstructionWeightsFirst);

			for (int j = 0; j < numReconstructionWeightsFirst; j++)
				qNode._connections[j]._weight = weightDist(generator);

			_qNodes.push_back(qNode);
		}
		else
			_input[i] = 0.0f;

	_actionPrev.clear();
	_actionPrev.assign(_actionNodes.size(), 0.0f);

	_maxActionPrev.clear();
	_maxActionPrev.assign(_actionNodes.size(), 0.0f);

	_hiddenStatesPrev.clear();
	_hiddenStatesPrev.assign(layerDescs.front()._spatialWidth * layerDescs.front()._spatialHeight, 0.0f);
}

void HTFERL::step(sys::ComputeSystem &cs, float reward, float qAlpha, float qGamma, float breakChance, float perturbationStdDev, float alphaQ, float alphaAction, float qTraceDecay, float actionTraceDecay, float actionTraceBeta, float actionTraceTemperature, int replayChainSize, int replayCount, std::mt19937 &generator) {
	std::uniform_int_distribution<int> seedDist(0, 99999);

	// ------------------------------------------------------------------------------
	// --------------------------------- Activate  ----------------------------------
	// ------------------------------------------------------------------------------

	for (int i = 0; i < _input.size(); i++)
		_htfe.setInput(i, _input[i]);

	_htfe.activate(cs, generator);

	// ------------------------------------------------------------------------------
	// ----------------------------- Q Value Updates  -------------------------------
	// ------------------------------------------------------------------------------

	Float2 layerOverInput;
	layerOverInput._x = static_cast<float>(_htfe.getLayerDescs().front()._spatialWidth - 1) / static_cast<float>(_htfe.getInputWidth() - 1);
	layerOverInput._y = static_cast<float>(_htfe.getLayerDescs().front()._spatialHeight - 1) / static_cast<float>(_htfe.getInputHeight() - 1);

	std::vector<Float2> firstHiddenVerbose(_htfe.getLayerDescs().front()._spatialWidth * _htfe.getLayerDescs().front()._spatialHeight);

	// Exploratory action
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, perturbationStdDev);

	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _htfe.getLayerDescs().front()._spatialWidth;
		region[1] = _htfe.getLayerDescs().front()._spatialHeight;
		region[2] = 1;

		cs.getQueue().enqueueReadImage(_htfe.getLayers().front()._hiddenStatesSpatial, CL_TRUE, origin, region, 0, 0, firstHiddenVerbose.data());
	}

	std::vector<float> firstHidden(_htfe.getLayerDescs().front()._spatialWidth * _htfe.getLayerDescs().front()._spatialHeight);

	for (int i = 0; i < firstHidden.size(); i++)
		firstHidden[i] = firstHiddenVerbose[i]._x;

	// Find nextQ
	float nextQ = 0.0f;
	float predQ = 0.0f;

	for (int i = 0; i < _qNodes.size(); i++) {
		float sum = 0.0f;

		int cx = std::round((_qNodes[i]._index % _htfe.getInputWidth()) * layerOverInput._x);
		int cy = std::round((_qNodes[i]._index / _htfe.getInputWidth()) * layerOverInput._y);

		int wi = 0;

		for (int dx = -_actionQRadius; dx <= _actionQRadius; dx++)
			for (int dy = -_actionQRadius; dy <= _actionQRadius; dy++) {
				int x = cx + dx;
				int y = cy + dy;

				if (x >= 0 && y >= 0 && x < _htfe.getLayerDescs().front()._spatialWidth && y < _htfe.getLayerDescs().front()._spatialHeight) {
					int j = x + y * _htfe.getLayerDescs().front()._spatialWidth;

					sum += _qNodes[i]._connections[wi]._weight * firstHidden[j];
				}

				wi++;
			}

		_qNodes[i]._output = sum;

		nextQ += _qNodes[i]._output;
		predQ += _htfe.getPrediction(_qNodes[i]._index);
	}

	nextQ /= _qNodes.size();
	predQ /= _qNodes.size();

	float tdError = reward + qGamma * nextQ - _prevValue;

	float newQ = _prevValue + qAlpha * tdError;

	// Update Q nodes
	for (int i = 0; i < _qNodes.size(); i++) {
		int cx = std::round((_qNodes[i]._index % _htfe.getInputWidth()) * layerOverInput._x);
		int cy = std::round((_qNodes[i]._index / _htfe.getInputWidth()) * layerOverInput._y);

		int wi = 0;

		for (int dx = -_actionQRadius; dx <= _actionQRadius; dx++)
			for (int dy = -_actionQRadius; dy <= _actionQRadius; dy++) {
				int x = cx + dx;
				int y = cy + dy;

				if (x >= 0 && y >= 0 && x < _htfe.getLayerDescs().front()._spatialWidth && y < _htfe.getLayerDescs().front()._spatialHeight) {
					int j = x + y * _htfe.getInputWidth();

					_qNodes[i]._connections[wi]._weight += alphaQ * tdError * _qNodes[i]._connections[wi]._trace;
					_qNodes[i]._connections[wi]._trace = std::max<float>((1.0f - qTraceDecay) * _qNodes[i]._connections[wi]._trace, firstHidden[j]);
				}

				wi++;
			}
	}

	// Retrieve action and then update actions
	for (int i = 0; i < _actionNodes.size(); i++) {
		float sum = 0.0f;

		int cx = std::round((_actionNodes[i]._index % _htfe.getInputWidth()) * layerOverInput._x);
		int cy = std::round((_actionNodes[i]._index / _htfe.getInputWidth()) * layerOverInput._y);

		int wi = 0;

		for (int dx = -_actionQRadius; dx <= _actionQRadius; dx++)
			for (int dy = -_actionQRadius; dy <= _actionQRadius; dy++) {
				int x = cx + dx;
				int y = cy + dy;

				if (x >= 0 && y >= 0 && x < _htfe.getLayerDescs().front()._spatialWidth && y < _htfe.getLayerDescs().front()._spatialHeight) {
					int j = x + y * _htfe.getInputWidth();

					sum += _actionNodes[i]._connections[wi]._weight * firstHidden[j];
				}

				wi++;
			}

		_actionNodes[i]._maxOutput = sum;

		if (dist01(generator) < breakChance)
			_actionNodes[i]._output = dist01(generator) * 2.0f - 1.0f;
		else
			_actionNodes[i]._output = std::min<float>(1.0f, std::max<float>(-1.0f, std::min<float>(1.0f, std::max<float>(-1.0f, _actionNodes[i]._maxOutput)) + pertDist(generator)));

		// Weight update and add to traces
		wi = 0;

		float outputError = _actionNodes[i]._output - _actionNodes[i]._maxOutput;

		float learn = std::max<float>(0.0f, tdError);

		for (int dx = -_actionQRadius; dx <= _actionQRadius; dx++)
			for (int dy = -_actionQRadius; dy <= _actionQRadius; dy++) {
				int x = cx + dx;
				int y = cy + dy;

				if (x >= 0 && y >= 0 && x < _htfe.getLayerDescs().front()._spatialWidth && y < _htfe.getLayerDescs().front()._spatialHeight) {
					int j = x + y * _htfe.getInputWidth();

					_actionNodes[i]._connections[wi]._weight += alphaAction * learn * _actionNodes[i]._connections[wi]._trace;
					_actionNodes[i]._connections[wi]._trace = (1.0f - actionTraceDecay) * _actionNodes[i]._connections[wi]._trace + std::exp(-std::abs(_actionNodes[i]._connections[wi]._trace) * actionTraceTemperature) * outputError * firstHidden[j];
				}

				wi++;
			}
	}

	std::cout << nextQ << " " << tdError << " " << predQ << " " << (nextQ - predQ) << std::endl;

	// ------------------------------------------------------------------------------
	// ---------------------- Weight Update and Predictions  ------------------------
	// ------------------------------------------------------------------------------

	_htfe.learn(cs);

	_prevValue = nextQ;

	for (int i = 0; i < _qNodes.size(); i++)
		_input[_qNodes[i]._index] = nextQ;

	for (int i = 0; i < _actionNodes.size(); i++)
		_input[_actionNodes[i]._index] = _actionNodes[i]._output;

	for (int i = 0; i < _actionNodes.size(); i++) {
		_actionPrev[i] = _actionNodes[i]._output;
		_maxActionPrev[i] = std::min<float>(1.0f, std::max<float>(-1.0f, _actionNodes[i]._maxOutput));
	}

	_hiddenStatesPrev = firstHidden;

	_htfe.stepEnd();
}

void HTFERL::exportStateData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const {
	std::mt19937 generator(seed);

	int maxWidth = _htfe.getInputWidth();
	int maxHeight = _htfe.getInputHeight();

	for (int l = 0; l < _htfe.getLayerDescs().size(); l++) {
		maxWidth = std::max<int>(maxWidth, _htfe.getLayerDescs()[l]._spatialWidth);
		maxHeight = std::max<int>(maxHeight, _htfe.getLayerDescs()[l]._spatialHeight);
	}

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::O)) {
			std::vector<float> state(_htfe.getInputWidth() * _htfe.getInputHeight());

			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _htfe.getInputWidth();
			region[1] = _htfe.getInputHeight();
			region[2] = 1;

			cs.getQueue().enqueueReadImage(_htfe.getInputImage(), CL_TRUE, origin, region, 0, 0, &state[0]);

			sf::Color c;
			c.r = uniformDist(generator) * 255.0f;
			c.g = uniformDist(generator) * 255.0f;
			c.b = uniformDist(generator) * 255.0f;

			// Convert to colors
			std::shared_ptr<sf::Image> image = std::make_shared<sf::Image>();

			image->create(maxWidth, maxHeight, sf::Color::Transparent);

			for (int x = 0; x < _htfe.getInputWidth(); x++)
				for (int y = 0; y < _htfe.getInputHeight(); y++) {
					sf::Color color;

					color = c;

					color.a = std::min<float>(1.0f, std::max<float>(0.0f, state[(x + y * _htfe.getInputWidth())])) * (255.0f - 3.0f) + 3;

					image->setPixel(x - _htfe.getInputWidth() / 2 + maxWidth / 2, y - _htfe.getInputHeight() / 2 + maxHeight / 2, color);
				}

			images.push_back(image);
		}
		else {
			std::vector<float> state(_htfe.getInputWidth() * _htfe.getInputHeight());

			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _htfe.getInputWidth();
			region[1] = _htfe.getInputHeight();
			region[2] = 1;

			cs.getQueue().enqueueReadImage(_htfe.getLayers().front()._predictedInputReconstruction, CL_TRUE, origin, region, 0, 0, &state[0]);

			sf::Color c;
			c.r = uniformDist(generator) * 255.0f;
			c.g = uniformDist(generator) * 255.0f;
			c.b = uniformDist(generator) * 255.0f;

			// Convert to colors
			std::shared_ptr<sf::Image> image = std::make_shared<sf::Image>();

			image->create(maxWidth, maxHeight, sf::Color::Transparent);

			for (int x = 0; x < _htfe.getInputWidth(); x++)
				for (int y = 0; y < _htfe.getInputHeight(); y++) {
					sf::Color color;

					color = c;

					color.a = std::min<float>(1.0f, std::max<float>(0.0f, state[(x + y * _htfe.getInputWidth())])) * (255.0f - 3.0f) + 3;

					image->setPixel(x - _htfe.getInputWidth() / 2 + maxWidth / 2, y - _htfe.getInputHeight() / 2 + maxHeight / 2, color);
				}

			images.push_back(image);
		}
	}
	else {
		for (int l = 0; l < _htfe.getLayerDescs().size(); l++) {
			std::vector<float> state(2 * _htfe.getLayerDescs()[l]._spatialWidth * _htfe.getLayerDescs()[l]._spatialHeight);

			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _htfe.getLayerDescs()[l]._spatialWidth;
			region[1] = _htfe.getLayerDescs()[l]._spatialHeight;
			region[2] = 1;

			cs.getQueue().enqueueReadImage(_htfe.getLayers()[l]._hiddenStatesSpatial, CL_TRUE, origin, region, 0, 0, &state[0]);

			sf::Color c;
			c.r = uniformDist(generator) * 255.0f;
			c.g = uniformDist(generator) * 255.0f;
			c.b = uniformDist(generator) * 255.0f;

			// Convert to colors
			std::shared_ptr<sf::Image> image = std::make_shared<sf::Image>();

			image->create(maxWidth, maxHeight, sf::Color::Transparent);

			for (int x = 0; x < _htfe.getLayerDescs()[l]._spatialWidth; x++)
				for (int y = 0; y < _htfe.getLayerDescs()[l]._spatialHeight; y++) {
					sf::Color color;

					color = c;

					color.a = std::min<float>(1.0f, std::max<float>(0.0f, state[2 * (x + y * _htfe.getLayerDescs()[l]._spatialWidth)])) * (255.0f - 3.0f) + 3;

					image->setPixel(x - _htfe.getLayerDescs()[l]._spatialWidth / 2 + maxWidth / 2, y - _htfe.getLayerDescs()[l]._spatialHeight / 2 + maxHeight / 2, color);
				}

			images.push_back(image);
		}
	}
}