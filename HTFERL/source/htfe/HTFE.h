#pragma once

#include "../system/ComputeSystem.h"
#include "../system/ComputeProgram.h"

#include <vector>
#include <list>

#include <random>

#include <memory>

namespace htfe {
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
		float _feedBackAlpha;
		float _hiddenBiasAlpha;
		float _reconstructionAlpha;
		float _feedForwardMomentum;
		float _lateralMomentum;
		float _feedBackMomentum;
		float _hiddenBiasMomentum;
		float _reconstructionMomentum;
		float _lateralScalar;
		float _feedBackScalar;
		float _minDerivative;
		float _blurKernelWidth;
		int _numBlurPasses;

		LayerDesc()
			: _width(16), _height(16), _receptiveFieldRadius(4), _reconstructionRadius(6), _lateralConnectionRadius(5), _inhibitionRadius(4), _feedBackConnectionRadius(6),
			_sparsity(3.01f / 81.0f), _dutyCycleDecay(0.01f),
			_feedForwardAlpha(0.01f), _lateralAlpha(0.05f), _feedBackAlpha(0.1f), _hiddenBiasAlpha(0.01f), _reconstructionAlpha(0.01f),
			_feedForwardMomentum(0.5f), _lateralMomentum(0.5f), _feedBackMomentum(0.5f), _hiddenBiasMomentum(0.5f), _reconstructionMomentum(0.5f),
			_lateralScalar(0.05f), _feedBackScalar(0.05f), _minDerivative(0.001f), _blurKernelWidth(1.0f), _numBlurPasses(1)
		{}
	};

	struct Layer {
		cl::Image2D _hiddenFeedForwardActivations;
		cl::Image2D _hiddenFeedBackActivations;
		cl::Image2D _hiddenFeedBackActivationsPrev;

		cl::Image2D _hiddenStatesFeedForward;
		cl::Image2D _hiddenStatesFeedForwardPrev;

		cl::Image2D _hiddenStatesFeedBack;
		cl::Image2D _hiddenStatesFeedBackPrev;
		cl::Image2D _hiddenStatesFeedBackPrevPrev;

		cl::Image2D _blurPing;
		cl::Image2D _blurPong;

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

	class HTFE {
	private:
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
		cl::Kernel _gaussianBlurXKernel;
		cl::Kernel _gaussianBlurYKernel;

		std::vector<float> _input;
		std::vector<float> _prediction;

		cl::Image2D _inputImage;
		cl::Image2D _inputImagePrev;

		void gaussianBlur(sys::ComputeSystem &cs, cl::Image2D &source, cl::Image2D &ping, cl::Image2D &pong, int imageSizeX, int imageSizeY, int passes, float kernelWidth);

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, float minInitWeight, float maxInitWeight);

		void activate(sys::ComputeSystem &cs);
		void learn(sys::ComputeSystem &cs);
		void stepEnd();

		int getInputWidth() const {
			return _inputWidth;
		}

		int getInputHeight() const {
			return _inputHeight;
		}

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		const std::vector<Layer> &getLayers() const {
			return _layers;
		}

		const cl::Image2D &getInputImage() const {
			return _inputImage;
		}

		void setInput(int i, float value) {
			_input[i] = value;
		}

		void setInput(int x, int y, float value) {
			setInput(x + y * _inputWidth, value);
		}

		float getPrediction(int i) const {
			return _prediction[i];
		}

		float getPrediction(int x, int y) const {
			return getPrediction(x + y * _inputWidth);
		}

		void clearMemory(sys::ComputeSystem &cs);
	};
}