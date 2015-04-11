#pragma once

#include "../system/ComputeSystem.h"
#include "../system/ComputeProgram.h"

#include <vector>
#include <list>

#include <random>

#include <memory>

namespace htfe {
	struct LayerDesc {
		int _spatialWidth, _spatialHeight;
		int _temporalWidth, _temporalHeight;

		int _receptiveFieldRadius;
		int _reconstructionRadius;
		int _predictiveRadius;
		int _lateralConnectionRadius;
		int _spatialInhibitionRadius;
		int _temporalInhibitionRadius;
		int _feedBackConnectionRadius;

		float _spatialSparsity;
		float _spatialLifetimeSparsity;
		float _temporalSparsity;
		float _temporalLifetimeSparsity;

		float _dutyCycleDecay;

		float _spatialAlpha;
		float _predictiveAlpha;
		float _lateralAlpha;
		float _feedBackAlpha;
		float _reconstructionAlpha;

		float _spatialLambda;
		float _temporalLambda;

		float _spatialMomentum;
		float _predictiveMomentum;
		float _lateralMomentum;
		float _feedBackMomentum;
		float _reconstructionMomentum;
		float _lateralScalar;
		float _feedBackScalar;
		float _blurKernelWidth;
		int _numBlurPasses;

		float _gaussianNoise;

		float _averageActivationDecay;
		float _lifetimeSparsityCorrectionFactor;
		float _boostIntensity;
		float _boostRatio;

		float _noise;

		LayerDesc()
			: _spatialWidth(16), _spatialHeight(16), _temporalWidth(16), _temporalHeight(16),
			_receptiveFieldRadius(6), _reconstructionRadius(6), _predictiveRadius(5), _lateralConnectionRadius(6), _spatialInhibitionRadius(5), _temporalInhibitionRadius(5), _feedBackConnectionRadius(6),
			_spatialSparsity(1.01f / 121.0f), _spatialLifetimeSparsity(1.01f / 121.0f), _temporalSparsity(1.01f / 121.0f), _temporalLifetimeSparsity(1.01f / 121.0f), _dutyCycleDecay(0.0002f),
			_spatialAlpha(0.5f), _predictiveAlpha(0.5f), _lateralAlpha(0.5f), _feedBackAlpha(0.5f), _reconstructionAlpha(0.15f),
			_spatialLambda(0.5f), _temporalLambda(0.5f),
			_spatialMomentum(0.0f), _predictiveMomentum(0.0f), _lateralMomentum(0.0f), _feedBackMomentum(0.0f), _reconstructionMomentum(0.0f),
			_lateralScalar(1.0f), _feedBackScalar(1.0f), _blurKernelWidth(1.0f), _numBlurPasses(0), _gaussianNoise(0.05f),
			_averageActivationDecay(0.001f), _lifetimeSparsityCorrectionFactor(0.2f), _boostIntensity(5.0f), _boostRatio(0.1f), _noise(0.001f)
		{}
	};

	struct Layer {
		cl::Image2D _hiddenActivationsSpatial;
		cl::Image2D _hiddenStatesSpatial;
		cl::Image2D _hiddenStatesSpatialPrev;

		cl::Image2D _hiddenActivationsTemporal;
		cl::Image2D _hiddenStatesTemporal;
		cl::Image2D _hiddenStatesTemporalPrev;
		cl::Image2D _hiddenStatesTemporalPrevPrev;

		cl::Image3D _spatialWeights;
		cl::Image3D _spatialWeightsPrev;

		cl::Image3D _spatialPredictiveReconstructionWeights;
		cl::Image3D _spatialPredictiveReconstructionWeightsPrev;

		cl::Image3D _predictiveWeights;
		cl::Image3D _predictiveWeightsPrev;

		cl::Image3D _lateralWeights;
		cl::Image3D _lateralWeightsPrev;

		cl::Image3D _feedBackWeights;
		cl::Image3D _feedBackWeightsPrev;

		cl::Image2D _predictedSpatial;
		cl::Image2D _predictedSpatialPrev;

		cl::Image2D _predictedInputReconstruction;
	};

	class HTFE {
	private:
		int _inputWidth, _inputHeight;

		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		cl::Kernel _layerInhibitKernel;
		cl::Kernel _layerHiddenStatesSpatialActivateKernel;
		cl::Kernel _layerHiddenStatesTemporalActivateKernel;
		cl::Kernel _layerHiddenStatesTemporalActivateLastKernel;
		cl::Kernel _layerInputReconstructKernel;
		cl::Kernel _layerUpdateSpatialWeightsKernel;
		cl::Kernel _layerSpatialPredictiveReconstructKernel;
		cl::Kernel _layerUpdateTemporalWeightsKernel;
		cl::Kernel _layerUpdateTemporalWeightsLastKernel;
		cl::Kernel _layerSpatialPredictiveReconstructionWeightUpdateKernel;
		cl::Kernel _gaussianBlurXKernel;
		cl::Kernel _gaussianBlurYKernel;

		std::vector<float> _input;
		std::vector<float> _prediction;

		cl::Image2D _inputImage;
		cl::Image2D _inputImagePrev;

		void gaussianBlur(sys::ComputeSystem &cs, cl::Image2D &source, cl::Image2D &ping, cl::Image2D &pong, int imageSizeX, int imageSizeY, int passes, float kernelWidth);

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, float minInitWeight, float maxInitWeight);

		void activate(sys::ComputeSystem &cs, std::mt19937 &generator);
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
	};
}