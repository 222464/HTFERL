#include "HTFERL.h"

#include <iostream>

using namespace htferl;

void HTFERL::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<htfe::LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};

	std::uniform_int_distribution<int> seedDist(0, 99999);

	_htfe.createRandom(cs, program, inputWidth, inputHeight, layerDescs, minInitWeight, maxInitWeight);

	_inputTypes = inputTypes;

	std::uniform_real_distribution<float> weightDist(minInitWeight, maxInitWeight);
	std::uniform_real_distribution<float> actionDist(0.0f, 1.0f);

	_prevValue = 0.0f;

	cl::Kernel initializeLayerHiddenKernel = cl::Kernel(program.getProgram(), "initializeLayerHidden");
	cl::Kernel initializeLayerVisibleKernel = cl::Kernel(program.getProgram(), "initializeLayerVisible");

	_input.clear();
	_input.resize(inputWidth * inputHeight);

	int numReconstructionWeightsFirst = std::pow(layerDescs.front()._reconstructionRadius * 2 + 1, 2);

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
}

void HTFERL::step(sys::ComputeSystem &cs, float reward, float qAlpha, float qGamma, float breakChance, float perturbationStdDev, float alphaQ, float alphaAction, float traceDecay, std::mt19937 &generator) {
	struct Float2 {
		float _x, _y;
	};

	struct Float4 {
		float _x, _y, _z, _w;
	};

	struct Int2 {
		int _x, _y;
	};
	
	std::uniform_int_distribution<int> seedDist(0, 99999);

	// ------------------------------------------------------------------------------
	// --------------------------------- Activate  ----------------------------------
	// ------------------------------------------------------------------------------

	for (int i = 0; i < _input.size(); i++)
		_htfe.setInput(i, _input[i]);

	_htfe.activate(cs);

	// ------------------------------------------------------------------------------
	// ----------------------------- Q Value Updates  -------------------------------
	// ------------------------------------------------------------------------------

	Float2 layerOverInput;
	layerOverInput._x = static_cast<float>(_htfe.getLayerDescs().front()._width) / static_cast<float>(_htfe.getInputWidth());
	layerOverInput._y = static_cast<float>(_htfe.getLayerDescs().front()._height) / static_cast<float>(_htfe.getInputHeight());

	std::vector<float> firstHidden(_htfe.getLayerDescs().front()._width * _htfe.getLayerDescs().front()._height);

	// Exploratory action
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, perturbationStdDev);

	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _htfe.getLayerDescs().front()._width;
		region[1] = _htfe.getLayerDescs().front()._height;
		region[2] = 1;

		cs.getQueue().enqueueReadImage(_htfe.getLayers().front()._hiddenStatesFeedBack, CL_TRUE, origin, region, 0, 0, firstHidden.data());
	}

	// Find nextQ
	float nextQ = 0.0f;

	for (int i = 0; i < _qNodes.size(); i++) {
		float sum = 0.0f;

		int cx = std::round((_qNodes[i]._index % _htfe.getInputWidth() + 1) * layerOverInput._x) - 1;
		int cy = std::round((_qNodes[i]._index / _htfe.getInputWidth() + 1) * layerOverInput._y) - 1;

		int wi = 0;

		for (int dx = -_htfe.getLayerDescs().front()._reconstructionRadius; dx <= _htfe.getLayerDescs().front()._reconstructionRadius; dx++)
			for (int dy = -_htfe.getLayerDescs().front()._reconstructionRadius; dy <= _htfe.getLayerDescs().front()._reconstructionRadius; dy++) {
				int x = cx + dx;
				int y = cy + dy;

				if (x >= 0 && y >= 0 && x < _htfe.getLayerDescs().front()._width && y < _htfe.getLayerDescs().front()._height) {
					int j = x + y * _htfe.getLayerDescs().front()._width;

					sum += _qNodes[i]._connections[wi]._weight * firstHidden[j];
				}

				wi++;
			}

		_qNodes[i]._output = sum;

		nextQ += _qNodes[i]._output;
	}

	nextQ /= _qNodes.size();

	float tdError = qAlpha * (reward + qGamma * nextQ - _prevValue);

	float newQ = _prevValue + tdError;

	// Update Q
	for (int i = 0; i < _qNodes.size(); i++) {
		int cx = std::round((_qNodes[i]._index % _htfe.getInputWidth() + 1) * layerOverInput._x) - 1;
		int cy = std::round((_qNodes[i]._index / _htfe.getInputWidth() + 1) * layerOverInput._y) - 1;

		int wi = 0;

		for (int dx = -_htfe.getLayerDescs().front()._reconstructionRadius; dx <= _htfe.getLayerDescs().front()._reconstructionRadius; dx++)
			for (int dy = -_htfe.getLayerDescs().front()._reconstructionRadius; dy <= _htfe.getLayerDescs().front()._reconstructionRadius; dy++) {
				int x = cx + dx;
				int y = cy + dy;

				if (x >= 0 && y >= 0 && x < _htfe.getLayerDescs().front()._width && y < _htfe.getLayerDescs().front()._height) {
					int j = x + y * _htfe.getLayerDescs().front()._width;

					_qNodes[i]._connections[wi]._weight += alphaQ * tdError * _qNodes[i]._connections[wi]._trace;
					_qNodes[i]._connections[wi]._trace = std::max<float>((1.0f - traceDecay) * _qNodes[i]._connections[wi]._trace, firstHidden[j]);
				}

				wi++;
			}
	}

	// ---------------------------------- Retrieve action ----------------------------------

	for (int i = 0; i < _actionNodes.size(); i++) {
		float sum = 0.0f;

		int cx = std::round((_actionNodes[i]._index % _htfe.getInputWidth() + 1) * layerOverInput._x) - 1;
		int cy = std::round((_actionNodes[i]._index / _htfe.getInputWidth() + 1) * layerOverInput._y) - 1;

		int wi = 0;

		for (int dx = -_htfe.getLayerDescs().front()._reconstructionRadius; dx <= _htfe.getLayerDescs().front()._reconstructionRadius; dx++)
			for (int dy = -_htfe.getLayerDescs().front()._reconstructionRadius; dy <= _htfe.getLayerDescs().front()._reconstructionRadius; dy++) {
				int x = cx + dx;
				int y = cy + dy;

				if (x >= 0 && y >= 0 && x < _htfe.getLayerDescs().front()._width && y < _htfe.getLayerDescs().front()._height) {
					int j = x + y * _htfe.getLayerDescs().front()._width;

					sum += _actionNodes[i]._connections[wi]._weight * firstHidden[j];
				}

				wi++;
			}

		_actionNodes[i]._maxOutput = sum;// std::min<float>(1.0f, std::max<float>(-1.0f, sum));

		if (dist01(generator) < breakChance)
			_actionNodes[i]._output = dist01(generator) * 2.0f - 1.0f;
		else
			_actionNodes[i]._output = std::min<float>(1.0f, std::max<float>(-1.0f, _actionNodes[i]._maxOutput + pertDist(generator)));
	}

	// ----------------------------------- Update Action -----------------------------------

	float aAlpha = alphaAction * tdError;

	for (int i = 0; i < _actionNodes.size(); i++) {
		int cx = std::round((_actionNodes[i]._index % _htfe.getInputWidth() + 1) * layerOverInput._x) - 1;
		int cy = std::round((_actionNodes[i]._index / _htfe.getInputWidth() + 1) * layerOverInput._y) - 1;

		int wi = 0;

		for (int dx = -_htfe.getLayerDescs().front()._reconstructionRadius; dx <= _htfe.getLayerDescs().front()._reconstructionRadius; dx++)
			for (int dy = -_htfe.getLayerDescs().front()._reconstructionRadius; dy <= _htfe.getLayerDescs().front()._reconstructionRadius; dy++) {
				int x = cx + dx;
				int y = cy + dy;

				if (x >= 0 && y >= 0 && x < _htfe.getLayerDescs().front()._width && y < _htfe.getLayerDescs().front()._height) {
					int j = x + y * _htfe.getLayerDescs().front()._width;

					_actionNodes[i]._connections[wi]._weight += aAlpha * _actionNodes[i]._connections[wi]._trace;
					_actionNodes[i]._connections[wi]._trace = (1.0f - traceDecay) * _actionNodes[i]._connections[wi]._trace + (_actionNodes[i]._output - _actionNodes[i]._maxOutput) * firstHidden[j];
				}

				wi++;
			}
	}

	_prevValue = nextQ;

	std::cout << nextQ << " " << tdError << " " << _actionNodes[0]._output << std::endl;

	// ------------------------------------------------------------------------------
	// ---------------------- Weight Update and Predictions  ------------------------
	// ------------------------------------------------------------------------------

	_htfe.learn(cs);

	// ------------------------------------------------------------------------------
	// -------------------------------- Update Input --------------------------------
	// ------------------------------------------------------------------------------

	for (int i = 0; i < _qNodes.size(); i++)
		_input[_qNodes[i]._index] = newQ;

	for (int i = 0; i < _actionNodes.size(); i++)
		_input[_actionNodes[i]._index] = _actionNodes[i]._output;

	_htfe.stepEnd();
}

void HTFERL::exportStateData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const {
	std::mt19937 generator(seed);
	
	int maxWidth = _htfe.getInputWidth();
	int maxHeight = _htfe.getInputHeight();

	for (int l = 0; l < _htfe.getLayerDescs().size(); l++) {
		maxWidth = std::max<int>(maxWidth, _htfe.getLayerDescs()[l]._width);
		maxHeight = std::max<int>(maxHeight, _htfe.getLayerDescs()[l]._height);
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

			cs.getQueue().enqueueReadImage(_htfe.getLayers().front()._visibleReconstruction, CL_TRUE, origin, region, 0, 0, &state[0]);

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
			std::vector<float> state(_htfe.getLayerDescs()[l]._width * _htfe.getLayerDescs()[l]._height);

			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _htfe.getLayerDescs()[l]._width;
			region[1] = _htfe.getLayerDescs()[l]._height;
			region[2] = 1;

			cs.getQueue().enqueueReadImage(_htfe.getLayers()[l]._hiddenStatesFeedBack, CL_TRUE, origin, region, 0, 0, &state[0]);

			sf::Color c;
			c.r = uniformDist(generator) * 255.0f;
			c.g = uniformDist(generator) * 255.0f;
			c.b = uniformDist(generator) * 255.0f;

			// Convert to colors
			std::shared_ptr<sf::Image> image = std::make_shared<sf::Image>();

			image->create(maxWidth, maxHeight, sf::Color::Transparent);

			for (int x = 0; x < _htfe.getLayerDescs()[l]._width; x++)
				for (int y = 0; y < _htfe.getLayerDescs()[l]._height; y++) {
					sf::Color color;

					color = c;

					color.a = std::min<float>(1.0f, std::max<float>(0.0f, state[(x + y * _htfe.getLayerDescs()[l]._width)])) * (255.0f - 3.0f) + 3;

					image->setPixel(x - _htfe.getLayerDescs()[l]._width / 2 + maxWidth / 2, y - _htfe.getLayerDescs()[l]._height / 2 + maxHeight / 2, color);
				}

			images.push_back(image);
		}
	}
}