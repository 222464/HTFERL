#pragma once

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <SFML/Graphics.hpp>

#include <vector>
#include <list>

#include <random>

#include <memory>

namespace htferl {
	class HTFERL {
	public:
		enum InputType {
			_state, _action
		};

		struct LayerDesc {
			int _width, _height;

			int _receptiveFieldRadius;
			int _reconstructionRadius;
			int _lateralConnectionRadius;
			int _inhibitionRadius;
			int _feedBackConnectionRadius;

			float _sparsity;

			float _dutyCycleDecay;
			float _alpha;
			float _beta;
			float _traceDecay;

			LayerDesc()
				: _width(16), _height(16), _receptiveFieldRadius(3), _reconstructionRadius(3), _lateralConnectionRadius(4), _inhibitionRadius(2), _feedBackConnectionRadius(4),
				_sparsity(0.05f), _dutyCycleDecay(0.01f), _alpha(0.1f), _beta(0.1f), _traceDecay(1.0f)
			{}
		};

	private:
		struct Layer {
			cl::Image2D _hiddenFeedForwardActivations;
			cl::Image2D _hiddenFeedBackActivations;

			cl::Image2D _hiddenStates;
			cl::Image2D _hiddenStatesPrev;
			cl::Image2D _hiddenStatesPrevPrev;

			cl::Image3D _feedForwardWeights;
			cl::Image3D _feedForwardWeightsPrev;

			cl::Image3D _reconstructionWeights;
			cl::Image3D _reconstructionWeightsPrev;

			cl::Image2D _visibleBiases;
			cl::Image2D _visibleBiasesPrev;

			cl::Image2D _hiddenBiases;
			cl::Image2D _hiddenBiasesPrev;

			cl::Image3D _lateralWeights;
			cl::Image3D _lateralWeightsPrev;

			cl::Image3D _feedBackWeights;
			cl::Image3D _feedBackWeightsPrev;

			cl::Image2D _visibleReconstruction;
			cl::Image2D _visibleReconstructionPrev;
		};

		int _inputWidth, _inputHeight;

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		cl::Kernel _layerHiddenFeedForwardActivateKernel;
		cl::Kernel _layerHiddenFeedBackActivateKernel;
		cl::Kernel _layerHiddenInhibitKernel;
		cl::Kernel _layerVisibleReconstructKernel;
		cl::Kernel _layerHiddenWeightUpdateKernel;
		cl::Kernel _layerVisibleWeightUpdateKernel;

		std::vector<float> _input;

		std::vector<InputType> _inputTypes;

		float _prevMax;
		float _prevValue;

		cl::Image2D _inputImage;

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, std::mt19937 &generator);
	
		void step(sys::ComputeSystem &cs, float reward, float alpha, float gamma, float breakChance, float perturbationStdDev, std::mt19937 &generator);

		int getInputWidth() const {
			return _inputWidth;
		}

		int getInputHeight() const {
			return _inputHeight;
		}

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		void setInput(int i, float value) {
			_input[i] = value;
		}

		void setInput(int x, int y, float value) {
			setInput(x + y * _inputWidth, value);
		}

		float getOutput(int i) const {
			return _input[i];
		}

		float getOutput(int x, int y) const {
			return getOutput(x + y * _inputWidth);
		}

		void exportStateData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const;
	};
}