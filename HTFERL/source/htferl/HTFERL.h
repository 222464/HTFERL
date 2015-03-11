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
			_state = 0, _action = 1, _q = 2
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
			float _gamma;
			float _lateralScalar;

			float _qWeight;

			LayerDesc()
				: _width(16), _height(16), _receptiveFieldRadius(3), _reconstructionRadius(4), _lateralConnectionRadius(3), _inhibitionRadius(2), _feedBackConnectionRadius(4),
				_sparsity(1.01f / 25.0f), _dutyCycleDecay(0.01f),
				_feedForwardAlpha(0.1f), _lateralAlpha(0.05f),
				_gamma(0.0f), _lateralScalar(0.25f)
			{}
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		struct Layer {
			cl::Image2D _hiddenFeedForwardActivations;
			cl::Image2D _hiddenFeedBackActivations;
			cl::Image2D _hiddenFeedBackActivationsPrev;

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
		};

		struct OutputConnection {
			float _weight;
			float _trace;

			OutputConnection()
				: _trace(0.0f)
			{}
		};

		struct QNode {
			float _output;
			int _index;
			
			std::vector<OutputConnection> _connections;

			QNode()
				: _output(0.0f)
			{}
		};

		struct ActionNode {
			float _output;
			float _maxOutput;
			int _index;

			std::vector<OutputConnection> _connections;

			ActionNode()
				: _output(0.0f), _maxOutput(0.0f)
			{}
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
		cl::Kernel _layerUpdateQKernel;

		std::vector<float> _input;

		std::vector<InputType> _inputTypes;

		std::vector<ActionNode> _actionNodes;
		std::vector<QNode> _qNodes;

		float _prevMax;
		float _prevValue;

		cl::Image2D _inputImage;
		cl::Image2D _inputImagePrev;

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, std::mt19937 &generator);
	
		void step(sys::ComputeSystem &cs, float reward, float qAlpha, float qGamma, float breakChance, float perturbationStdDev, float alphaQ, float alphaAction, float beta, float traceDecay, float temperature, std::mt19937 &generator);

		int getInputWidth() const {
			return _inputWidth;
		}

		int getInputHeight() const {
			return _inputHeight;
		}

		int getNumActions() const {
			return _actionNodes.size();
		}

		int getNumQNodes() const {
			return _qNodes.size();
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
			return _actionNodes[i]._output;
		}

		void exportStateData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const;
	};
}