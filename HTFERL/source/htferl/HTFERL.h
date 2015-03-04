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
			float _feedForwardAlpha;
			float _lateralAlpha;
			float _feedForwardBeta;
			float _lateralBeta;
			float _gamma;
			float _lateralScalar;
			float _traceDecay;

			LayerDesc()
				: _width(16), _height(16), _receptiveFieldRadius(3), _reconstructionRadius(3), _lateralConnectionRadius(3), _inhibitionRadius(3), _feedBackConnectionRadius(4),
				_sparsity(1.01f / 49.0f), _dutyCycleDecay(0.01f),
				_feedForwardAlpha(0.4f), _feedForwardBeta(0.5f), _lateralAlpha(0.2f),  _lateralBeta(0.5f),
				_gamma(0.005f), _lateralScalar(0.01f), _traceDecay(1.0f)
			{}
		};

	private:
		struct Layer {
			cl::Image2D _hiddenFeedForwardActivations;
			cl::Image2D _hiddenFeedBackActivations;

			cl::Image2D _hiddenStatesFeedForward;
			cl::Image2D _hiddenStatesFeedForwardPrev;

			cl::Image2D _hiddenStatesFeedBack;
			cl::Image2D _hiddenStatesFeedBackPrev;
			cl::Image2D _hiddenStatesFeedBackPrevPrev;

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

			cl::Image2D _qValues;
			cl::Image2D _qValuesPrev;
		};

		int _inputWidth, _inputHeight;

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		cl::Kernel _layerHiddenFeedForwardActivateKernel;
		cl::Kernel _layerHiddenFeedBackActivateKernel;
		cl::Kernel _layerHiddenInhibitKernel;
		cl::Kernel _layerVisibleReconstructKernel;
		cl::Kernel _layerHiddenWeightUpdateKernel;
		cl::Kernel _layerHiddenWeightUpdateLastKernel;
		cl::Kernel _layerVisibleWeightUpdateKernel;

		std::vector<float> _input;

		std::vector<InputType> _inputTypes;

		float _prevMax;
		float _prevValue;

		cl::Image2D _inputImage;
		cl::Image2D _inputImagePrev;

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