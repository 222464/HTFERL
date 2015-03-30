#include "HTFE.h"

#include <iostream>
#include <time.h>

using namespace htfe;

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

void HTFE::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, float minInitWeight, float maxInitWeight) {
	std::mt19937 generator(time(nullptr));

	std::uniform_int_distribution<int> seedDist(0, 99999);

	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	cl::Kernel initializeLayerHiddenKernel = cl::Kernel(program.getProgram(), "initializeLayerHidden");
	cl::Kernel initializeLayerVisibleKernel = cl::Kernel(program.getProgram(), "initializeLayerVisible");

	_input.clear();
	_input.resize(_inputWidth * _inputHeight, 0.0f);

	_prediction.clear();
	_prediction.resize(_inputWidth * _inputHeight, 0.0f);

	_inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);
	_inputImagePrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	{
		cl_uint4 clear = { 0, 0, 0, 0 };

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueFillImage(_inputImage, clear, origin, region);
		cs.getQueue().enqueueFillImage(_inputImagePrev, clear, origin, region);
	}

	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		int numFeedForwardWeights = std::pow(_layerDescs[l]._receptiveFieldRadius * 2 + 1, 2) + 1; // + 1 for bias
		int numReconstructionWeights = std::pow(_layerDescs[l]._reconstructionRadius * 2 + 1, 2) + 1; // + 1 for bias
		int numPredictiveWeights = std::pow(_layerDescs[l]._predictiveRadius * 2 + 1, 2) + 1; // + 1 for bias
		int numLateralWeights = std::pow(_layerDescs[l]._lateralConnectionRadius * 2 + 1, 2);
		int numFeedBackWeights = std::pow(_layerDescs[l]._feedBackConnectionRadius * 2 + 1, 2);

		_layers[l]._hiddenFeedForwardActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._hiddenFeedBackActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._hiddenFeedBackActivationsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._hiddenStatesFeedForward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._hiddenStatesFeedForwardPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._hiddenStatesFeedBack = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._hiddenStatesFeedBackPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._hiddenStatesFeedBackPrevPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._blurPing = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._blurPong = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._feedForwardWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numFeedForwardWeights);
		_layers[l]._feedForwardWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numFeedForwardWeights);

		_layers[l]._reconstructionWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), prevWidth, prevHeight, numReconstructionWeights);
		_layers[l]._reconstructionWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), prevWidth, prevHeight, numReconstructionWeights);

		_layers[l]._predictiveWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numPredictiveWeights);
		_layers[l]._predictiveWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numPredictiveWeights);

		_layers[l]._lateralWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numLateralWeights);
		_layers[l]._lateralWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numLateralWeights);

		_layers[l]._feedBackWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numFeedBackWeights);
		_layers[l]._feedBackWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numFeedBackWeights);

		_layers[l]._spatialReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevWidth, prevHeight);

		_layers[l]._visibleReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevWidth, prevHeight);
		_layers[l]._visibleReconstructionPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevWidth, prevHeight);

		// Initialize
		Uint2 initSeedHidden;
		initSeedHidden._x = seedDist(generator);
		initSeedHidden._y = seedDist(generator);

		int index = 0;

		initializeLayerHiddenKernel.setArg(index++, _layers[l]._hiddenFeedForwardActivations);
		initializeLayerHiddenKernel.setArg(index++, _layers[l]._hiddenFeedBackActivations);
		initializeLayerHiddenKernel.setArg(index++, _layers[l]._hiddenStatesFeedForward);
		initializeLayerHiddenKernel.setArg(index++, _layers[l]._feedForwardWeights);
		initializeLayerHiddenKernel.setArg(index++, _layers[l]._predictiveWeights);
		initializeLayerHiddenKernel.setArg(index++, _layers[l]._lateralWeights);
		initializeLayerHiddenKernel.setArg(index++, _layers[l]._feedBackWeights);
		initializeLayerHiddenKernel.setArg(index++, numFeedForwardWeights);
		initializeLayerHiddenKernel.setArg(index++, numPredictiveWeights);
		initializeLayerHiddenKernel.setArg(index++, numLateralWeights);
		initializeLayerHiddenKernel.setArg(index++, numFeedBackWeights);
		initializeLayerHiddenKernel.setArg(index++, initSeedHidden);
		initializeLayerHiddenKernel.setArg(index++, _layerDescs[l]._sparsity);
		initializeLayerHiddenKernel.setArg(index++, _layerDescs[l]._lateralScalar);
		initializeLayerHiddenKernel.setArg(index++, _layerDescs[l]._feedBackScalar);
		initializeLayerHiddenKernel.setArg(index++, minInitWeight);
		initializeLayerHiddenKernel.setArg(index++, maxInitWeight);

		cs.getQueue().enqueueNDRangeKernel(initializeLayerHiddenKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		Uint2 initSeedVisible;
		initSeedVisible._x = seedDist(generator);
		initSeedVisible._y = seedDist(generator);

		index = 0;

		initializeLayerVisibleKernel.setArg(index++, _layers[l]._visibleReconstruction);
		initializeLayerVisibleKernel.setArg(index++, _layers[l]._reconstructionWeights);
		initializeLayerVisibleKernel.setArg(index++, numReconstructionWeights);
		initializeLayerVisibleKernel.setArg(index++, initSeedVisible);
		initializeLayerVisibleKernel.setArg(index++, minInitWeight);
		initializeLayerVisibleKernel.setArg(index++, maxInitWeight);

		cs.getQueue().enqueueNDRangeKernel(initializeLayerVisibleKernel, cl::NullRange, cl::NDRange(prevWidth, prevHeight));

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = 1;

			cs.getQueue().enqueueCopyImage(_layers[l]._hiddenFeedBackActivations, _layers[l]._hiddenFeedBackActivationsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = prevWidth;
			region[1] = prevHeight;
			region[2] = 1;

			cs.getQueue().enqueueCopyImage(_layers[l]._visibleReconstruction, _layers[l]._visibleReconstructionPrev, origin, origin, region);
			cs.getQueue().enqueueCopyImage(_layers[l]._visibleReconstruction, _layers[l]._spatialReconstruction, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = 1;

			cs.getQueue().enqueueCopyImage(_layers[l]._hiddenStatesFeedForward, _layers[l]._hiddenStatesFeedForwardPrev, origin, origin, region);
			cs.getQueue().enqueueCopyImage(_layers[l]._hiddenStatesFeedForward, _layers[l]._hiddenStatesFeedBack, origin, origin, region);
			cs.getQueue().enqueueCopyImage(_layers[l]._hiddenStatesFeedForward, _layers[l]._hiddenStatesFeedBackPrev, origin, origin, region);
			cs.getQueue().enqueueCopyImage(_layers[l]._hiddenStatesFeedForward, _layers[l]._hiddenStatesFeedBackPrevPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = numFeedForwardWeights;

			cs.getQueue().enqueueCopyImage(_layers[l]._feedForwardWeights, _layers[l]._feedForwardWeightsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = numPredictiveWeights;

			cs.getQueue().enqueueCopyImage(_layers[l]._predictiveWeights, _layers[l]._predictiveWeightsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = numLateralWeights;

			cs.getQueue().enqueueCopyImage(_layers[l]._lateralWeights, _layers[l]._lateralWeightsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = numFeedBackWeights;

			cs.getQueue().enqueueCopyImage(_layers[l]._feedBackWeights, _layers[l]._feedBackWeightsPrev, origin, origin, region);
		}

		{
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = prevWidth;
			region[1] = prevHeight;
			region[2] = numReconstructionWeights;

			cs.getQueue().enqueueCopyImage(_layers[l]._reconstructionWeights, _layers[l]._reconstructionWeightsPrev, origin, origin, region);
		}

		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
	}

	_layerHiddenFeedForwardActivateKernel = cl::Kernel(program.getProgram(), "layerHiddenFeedForwardActivate");
	_layerHiddenFeedBackActivateKernel = cl::Kernel(program.getProgram(), "layerHiddenFeedBackActivate");
	_layerHiddenInhibitKernel = cl::Kernel(program.getProgram(), "layerHiddenInhibit");
	_layerSpatialReconstructKernel = cl::Kernel(program.getProgram(), "layerSpatialReconstruct");
	_layerUpdateFeedForwardWeightsKernel = cl::Kernel(program.getProgram(), "layerUpdateFeedForwardWeights");
	_layerVisibleReconstructKernel = cl::Kernel(program.getProgram(), "layerVisibleReconstruct");
	_layerHiddenWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerHiddenWeightUpdate");
	_layerHiddenWeightUpdateLastKernel = cl::Kernel(program.getProgram(), "layerHiddenWeightUpdateLast");
	_layerVisibleWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerVisibleWeightUpdate");
	_gaussianBlurXKernel = cl::Kernel(program.getProgram(), "gaussianBlurX");
	_gaussianBlurYKernel = cl::Kernel(program.getProgram(), "gaussianBlurY");
}

void HTFE::activate(sys::ComputeSystem &cs) {
	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueWriteImage(_inputImage, CL_TRUE, origin, region, 0, 0, _input.data());
	}

	std::uniform_int_distribution<int> seedDist(0, 99999);

	// ------------------------------------------------------------------------------
	// ------------------------------------ Go up -----------------------------------
	// ------------------------------------------------------------------------------

	cl::Image2D* pPrevLayer = &_inputImage;
	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		float localActivity = std::round(_layerDescs[l]._sparsity * std::pow(2 * _layerDescs[l]._inhibitionRadius + 1, 2));

		Int2 layerSize;
		layerSize._x = _layerDescs[l]._width;
		layerSize._y = _layerDescs[l]._height;

		Int2 layerSizeMinusOne;
		layerSizeMinusOne._x = _layerDescs[l]._width - 1;
		layerSizeMinusOne._y = _layerDescs[l]._height - 1;

		Float2 layerSizeMinusOneInv;
		layerSizeMinusOneInv._x = 1.0f / (_layerDescs[l]._width - 1);
		layerSizeMinusOneInv._y = 1.0f / (_layerDescs[l]._height - 1);

		Int2 inputSize;
		inputSize._x = prevWidth;
		inputSize._y = prevHeight;

		Int2 inputSizeMinusOne;
		inputSizeMinusOne._x = prevWidth - 1;
		inputSizeMinusOne._y = prevHeight - 1;

		Float2 inputSizeMinusOneInv;
		inputSizeMinusOneInv._x = 1.0f / (prevWidth - 1);
		inputSizeMinusOneInv._y = 1.0f / (prevHeight - 1);

		// -------------------------------- Activate --------------------------------

		int index = 0;

		_layerHiddenFeedForwardActivateKernel.setArg(index++, *pPrevLayer);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, _layers[l]._feedForwardWeightsPrev);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, _layers[l]._hiddenFeedForwardActivations);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, layerSize);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, layerSizeMinusOneInv);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, inputSize);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, inputSizeMinusOne);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, _layerDescs[l]._receptiveFieldRadius);

		cs.getQueue().enqueueNDRangeKernel(_layerHiddenFeedForwardActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		// ---------------------------------- Inhibit ---------------------------------

		index = 0;

		_layerHiddenInhibitKernel.setArg(index++, _layers[l]._hiddenFeedForwardActivations);
		_layerHiddenInhibitKernel.setArg(index++, _layers[l]._hiddenStatesFeedForward);
		_layerHiddenInhibitKernel.setArg(index++, layerSize);
		_layerHiddenInhibitKernel.setArg(index++, _layerDescs[l]._inhibitionRadius);
		_layerHiddenInhibitKernel.setArg(index++, localActivity);

		cs.getQueue().enqueueNDRangeKernel(_layerHiddenInhibitKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		// Blur the inhibited results
		if (_layerDescs[l]._numBlurPasses != 0) {
			gaussianBlur(cs, _layers[l]._hiddenStatesFeedForward, _layers[l]._blurPing, _layers[l]._blurPong, _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._numBlurPasses, _layerDescs[l]._blurKernelWidth);

			// Copy back to hiddenStatesFeedForward
			{
				cl::size_t<3> origin;
				origin[0] = 0;
				origin[1] = 0;
				origin[2] = 0;

				cl::size_t<3> region;
				region[0] = _layerDescs[l]._width;
				region[1] = _layerDescs[l]._height;
				region[2] = 1;

				cs.getQueue().enqueueCopyImage(_layers[l]._blurPong, _layers[l]._hiddenStatesFeedForward, origin, origin, region);
			}
		}

		pPrevLayer = &_layers[l]._hiddenStatesFeedForward;
		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;
	}

	// ------------------------------------------------------------------------------
	// -------------------------------- Go back down --------------------------------
	// ------------------------------------------------------------------------------

	for (int l = _layers.size() - 1; l >= 0; l--) {
		if (l > 0) {
			pPrevLayer = &_layers[l - 1]._hiddenStatesFeedForward;
			prevWidth = _layerDescs[l - 1]._width;
			prevHeight = _layerDescs[l - 1]._height;
		}
		else {
			pPrevLayer = &_inputImage;
			prevWidth = _inputWidth;
			prevHeight = _inputHeight;
		}

		float localActivity = std::round(_layerDescs[l]._sparsity * std::pow(2 * _layerDescs[l]._inhibitionRadius + 1, 2));

		Int2 layerSize;
		layerSize._x = _layerDescs[l]._width;
		layerSize._y = _layerDescs[l]._height;

		Int2 layerSizeMinusOne;
		layerSizeMinusOne._x = _layerDescs[l]._width - 1;
		layerSizeMinusOne._y = _layerDescs[l]._height - 1;

		Float2 layerSizeMinusOneInv;
		layerSizeMinusOneInv._x = 1.0f / (_layerDescs[l]._width - 1);
		layerSizeMinusOneInv._y = 1.0f / (_layerDescs[l]._height - 1);

		Int2 inputSize;
		inputSize._x = prevWidth;
		inputSize._y = prevHeight;

		Int2 inputSizeMinusOne;
		inputSizeMinusOne._x = prevWidth - 1;
		inputSizeMinusOne._y = prevHeight - 1;

		Float2 inputSizeMinusOneInv;
		inputSizeMinusOneInv._x = 1.0f / (prevWidth - 1);
		inputSizeMinusOneInv._y = 1.0f / (prevHeight - 1);

		Int2 nextSize;
		Int2 nextSizeMinusOne;

		if (l == _layers.size() - 1) {
			nextSize._x = nextSize._y = 1;
			nextSizeMinusOne._x = nextSizeMinusOne._y = 0;
		}
		else {
			nextSize._x = _layerDescs[l + 1]._width;
			nextSize._y = _layerDescs[l + 1]._height;
			nextSizeMinusOne._x = _layerDescs[l + 1]._width - 1;
			nextSizeMinusOne._y = _layerDescs[l + 1]._height - 1;
		}

		// -------------------------------- Activate --------------------------------

		int index = 0;

		if (l == _layers.size() - 1) {
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;

			cl::size_t<3> region;
			region[0] = _layerDescs[l]._width;
			region[1] = _layerDescs[l]._height;
			region[2] = 1;

			cs.getQueue().enqueueCopyImage(_layers[l]._hiddenFeedForwardActivations, _layers[l]._hiddenFeedBackActivations, origin, origin, region);
		}
		else {
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layers[l]._hiddenStatesFeedForward);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrev);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layers[l + 1]._hiddenStatesFeedBack);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layers[l]._predictiveWeightsPrev);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layers[l]._feedBackWeightsPrev);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layers[l]._lateralWeightsPrev);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layers[l]._hiddenFeedBackActivations);
			_layerHiddenFeedBackActivateKernel.setArg(index++, layerSize);
			_layerHiddenFeedBackActivateKernel.setArg(index++, layerSizeMinusOneInv);
			_layerHiddenFeedBackActivateKernel.setArg(index++, nextSize);
			_layerHiddenFeedBackActivateKernel.setArg(index++, nextSizeMinusOne);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layerDescs[l]._predictiveRadius);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layerDescs[l]._feedBackConnectionRadius);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layerDescs[l]._lateralConnectionRadius);

			cs.getQueue().enqueueNDRangeKernel(_layerHiddenFeedBackActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}

		// ---------------------------------- Inhibit ---------------------------------

		index = 0;

		_layerHiddenInhibitKernel.setArg(index++, _layers[l]._hiddenFeedBackActivations);
		_layerHiddenInhibitKernel.setArg(index++, _layers[l]._hiddenStatesFeedBack);
		_layerHiddenInhibitKernel.setArg(index++, layerSize);
		_layerHiddenInhibitKernel.setArg(index++, _layerDescs[l]._inhibitionRadius);
		_layerHiddenInhibitKernel.setArg(index++, localActivity);

		cs.getQueue().enqueueNDRangeKernel(_layerHiddenInhibitKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		// --------------------- Make Predictions (Reconstruction) ---------------------

		index = 0;

		_layerVisibleReconstructKernel.setArg(index++, _layers[l]._hiddenStatesFeedBack);
		_layerVisibleReconstructKernel.setArg(index++, _layers[l]._reconstructionWeightsPrev);
		_layerVisibleReconstructKernel.setArg(index++, _layers[l]._visibleReconstruction);
		_layerVisibleReconstructKernel.setArg(index++, _layerDescs[l]._reconstructionRadius);
		_layerVisibleReconstructKernel.setArg(index++, inputSizeMinusOne);
		_layerVisibleReconstructKernel.setArg(index++, inputSizeMinusOneInv);
		_layerVisibleReconstructKernel.setArg(index++, layerSize);
		_layerVisibleReconstructKernel.setArg(index++, layerSizeMinusOne);
		_layerVisibleReconstructKernel.setArg(index++, layerSizeMinusOneInv);

		cs.getQueue().enqueueNDRangeKernel(_layerVisibleReconstructKernel, cl::NullRange, cl::NDRange(prevWidth, prevHeight));
	}

	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueReadImage(_layers.front()._visibleReconstruction, CL_TRUE, origin, region, 0, 0, _prediction.data());
	}
}

void HTFE::learn(sys::ComputeSystem &cs) {
	// ------------------------------------------------------------------------------
	// ---------------------- Weight Update and Predictions  ------------------------
	// ------------------------------------------------------------------------------

	cl::Image2D* pPrevLayer = &_inputImage;
	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;

	cl::Image2D* pPrevLayerFeedForwardPrev = &_inputImagePrev;
	cl::Image2D* pPrevLayerFeedBackPrev = &_inputImagePrev;

	for (int l = 0; l < _layers.size(); l++) {
		float localActivity = std::round(_layerDescs[l]._sparsity * std::pow(2 * _layerDescs[l]._inhibitionRadius + 1, 2));

		Int2 layerSize;
		layerSize._x = _layerDescs[l]._width;
		layerSize._y = _layerDescs[l]._height;

		Int2 layerSizeMinusOne;
		layerSizeMinusOne._x = _layerDescs[l]._width - 1;
		layerSizeMinusOne._y = _layerDescs[l]._height - 1;

		Float2 layerSizeMinusOneInv;
		layerSizeMinusOneInv._x = 1.0f / (_layerDescs[l]._width - 1);
		layerSizeMinusOneInv._y = 1.0f / (_layerDescs[l]._height - 1);

		Int2 inputSize;
		inputSize._x = prevWidth;
		inputSize._y = prevHeight;

		Int2 inputSizeMinusOne;
		inputSizeMinusOne._x = prevWidth - 1;
		inputSizeMinusOne._y = prevHeight - 1;

		Float2 inputSizeMinusOneInv;
		inputSizeMinusOneInv._x = 1.0f / (prevWidth - 1);
		inputSizeMinusOneInv._y = 1.0f / (prevHeight - 1);

		Int2 reverseReconstructionReceptiveRadius;
		reverseReconstructionReceptiveRadius._x = std::ceil(static_cast<float>(prevWidth) / static_cast<float>(_layerDescs[l]._width) * static_cast<float>(_layerDescs[l]._reconstructionRadius));
		reverseReconstructionReceptiveRadius._y = std::ceil(static_cast<float>(prevHeight) / static_cast<float>(_layerDescs[l]._height) * static_cast<float>(_layerDescs[l]._reconstructionRadius));

		Int2 reverseReceptiveRadius;
		reverseReceptiveRadius._x = std::ceil(static_cast<float>(_layerDescs[l]._width) / static_cast<float>(prevWidth) * static_cast<float>(_layerDescs[l]._receptiveFieldRadius));
		reverseReceptiveRadius._y = std::ceil(static_cast<float>(_layerDescs[l]._height) / static_cast<float>(prevHeight)* static_cast<float>(_layerDescs[l]._receptiveFieldRadius));

		Int2 nextSize;
		Int2 nextSizeMinusOne;

		if (l == _layers.size() - 1) {
			nextSize._x = nextSize._y = 1;
			nextSizeMinusOne._x = nextSizeMinusOne._y = 0;
		}
		else {
			nextSize._x = _layerDescs[l + 1]._width;
			nextSize._y = _layerDescs[l + 1]._height;
			nextSizeMinusOne._x = _layerDescs[l + 1]._width - 1;
			nextSizeMinusOne._y = _layerDescs[l + 1]._height - 1;
		}

		// ------------------------------- Weight Updates -------------------------------

		int index = 0;

		// Spatial reconstruction and weight update

		_layerSpatialReconstructKernel.setArg(index++, _layers[l]._hiddenStatesFeedForwardPrev);
		_layerSpatialReconstructKernel.setArg(index++, _layers[l]._feedForwardWeightsPrev);
		_layerSpatialReconstructKernel.setArg(index++, _layers[l]._spatialReconstruction);
		_layerSpatialReconstructKernel.setArg(index++, _layerDescs[l]._receptiveFieldRadius);
		_layerSpatialReconstructKernel.setArg(index++, reverseReceptiveRadius);
		_layerSpatialReconstructKernel.setArg(index++, inputSizeMinusOne);
		_layerSpatialReconstructKernel.setArg(index++, inputSizeMinusOneInv);
		_layerSpatialReconstructKernel.setArg(index++, layerSize);
		_layerSpatialReconstructKernel.setArg(index++, layerSizeMinusOne);
		_layerSpatialReconstructKernel.setArg(index++, layerSizeMinusOneInv);

		cs.getQueue().enqueueNDRangeKernel(_layerSpatialReconstructKernel, cl::NullRange, cl::NDRange(prevWidth, prevHeight));

		index = 0;

		_layerUpdateFeedForwardWeightsKernel.setArg(index++, *pPrevLayer);
		_layerUpdateFeedForwardWeightsKernel.setArg(index++, _layers[l]._spatialReconstruction);
		_layerUpdateFeedForwardWeightsKernel.setArg(index++, _layers[l]._hiddenStatesFeedForwardPrev);
		_layerUpdateFeedForwardWeightsKernel.setArg(index++, _layers[l]._feedForwardWeightsPrev);
		_layerUpdateFeedForwardWeightsKernel.setArg(index++, _layers[l]._feedForwardWeights);
		_layerUpdateFeedForwardWeightsKernel.setArg(index++, layerSize);
		_layerUpdateFeedForwardWeightsKernel.setArg(index++, layerSizeMinusOneInv);
		_layerUpdateFeedForwardWeightsKernel.setArg(index++, inputSize);
		_layerUpdateFeedForwardWeightsKernel.setArg(index++, inputSizeMinusOne);
		_layerUpdateFeedForwardWeightsKernel.setArg(index++, _layerDescs[l]._receptiveFieldRadius);
		_layerUpdateFeedForwardWeightsKernel.setArg(index++, _layerDescs[l]._feedForwardAlpha);
		_layerUpdateFeedForwardWeightsKernel.setArg(index++, _layerDescs[l]._feedForwardMomentum);

		cs.getQueue().enqueueNDRangeKernel(_layerUpdateFeedForwardWeightsKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		// Temporal reconstruction and weight update
		
		Float4 alphas;
		alphas._x = _layerDescs[l]._predictiveAlpha;
		alphas._y = _layerDescs[l]._lateralAlpha;
		alphas._z = _layerDescs[l]._feedBackAlpha;
		alphas._w = _layerDescs[l]._hiddenBiasAlpha;

		Float4 momenta;
		momenta._x = _layerDescs[l]._predictiveMomentum;
		momenta._y = _layerDescs[l]._lateralMomentum;
		momenta._z = _layerDescs[l]._feedBackMomentum;
		momenta._w = _layerDescs[l]._hiddenBiasMomentum;

		index = 0;

		if (l == _layers.size() - 1) {
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._visibleReconstructionPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, *pPrevLayer);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, *pPrevLayerFeedForwardPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._hiddenStatesFeedForwardPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._hiddenFeedBackActivationsPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrevPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._reconstructionWeightsPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._predictiveWeightsPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._lateralWeightsPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._predictiveWeights);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._lateralWeights);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, layerSize);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, layerSizeMinusOne);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, layerSizeMinusOneInv);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, inputSize);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, inputSizeMinusOne);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, inputSizeMinusOneInv);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, reverseReconstructionReceptiveRadius);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layerDescs[l]._predictiveRadius);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layerDescs[l]._lateralConnectionRadius);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layerDescs[l]._reconstructionRadius);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layerDescs[l]._sparsity);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, alphas);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, momenta);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layerDescs[l]._minDerivative);

			cs.getQueue().enqueueNDRangeKernel(_layerHiddenWeightUpdateLastKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}
		else {
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._visibleReconstructionPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, *pPrevLayer);
			_layerHiddenWeightUpdateKernel.setArg(index++, *pPrevLayerFeedForwardPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._hiddenStatesFeedForwardPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._hiddenFeedBackActivationsPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrevPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l + 1]._hiddenStatesFeedBackPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._reconstructionWeightsPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._predictiveWeightsPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._lateralWeightsPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._feedBackWeightsPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._predictiveWeights);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._lateralWeights);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._feedBackWeights);
			_layerHiddenWeightUpdateKernel.setArg(index++, layerSize);
			_layerHiddenWeightUpdateKernel.setArg(index++, layerSizeMinusOne);
			_layerHiddenWeightUpdateKernel.setArg(index++, layerSizeMinusOneInv);
			_layerHiddenWeightUpdateKernel.setArg(index++, inputSize);
			_layerHiddenWeightUpdateKernel.setArg(index++, inputSizeMinusOne);
			_layerHiddenWeightUpdateKernel.setArg(index++, inputSizeMinusOneInv);
			_layerHiddenWeightUpdateKernel.setArg(index++, nextSize);
			_layerHiddenWeightUpdateKernel.setArg(index++, nextSizeMinusOne);
			_layerHiddenWeightUpdateKernel.setArg(index++, reverseReconstructionReceptiveRadius);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._predictiveRadius);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._lateralConnectionRadius);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._feedBackConnectionRadius);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._reconstructionRadius);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._sparsity);
			_layerHiddenWeightUpdateKernel.setArg(index++, alphas);
			_layerHiddenWeightUpdateKernel.setArg(index++, momenta);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._minDerivative);

			cs.getQueue().enqueueNDRangeKernel(_layerHiddenWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}

		index = 0;

		_layerVisibleWeightUpdateKernel.setArg(index++, _layers[l]._visibleReconstructionPrev);
		_layerVisibleWeightUpdateKernel.setArg(index++, *pPrevLayer);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrev);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layers[l]._reconstructionWeightsPrev);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layers[l]._reconstructionWeights);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layerDescs[l]._reconstructionRadius);
		_layerVisibleWeightUpdateKernel.setArg(index++, inputSizeMinusOne);
		_layerVisibleWeightUpdateKernel.setArg(index++, inputSizeMinusOneInv);
		_layerVisibleWeightUpdateKernel.setArg(index++, layerSize);
		_layerVisibleWeightUpdateKernel.setArg(index++, layerSizeMinusOne);
		_layerVisibleWeightUpdateKernel.setArg(index++, layerSizeMinusOneInv);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layerDescs[l]._reconstructionAlpha);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layerDescs[l]._reconstructionMomentum);

		cs.getQueue().enqueueNDRangeKernel(_layerVisibleWeightUpdateKernel, cl::NullRange, cl::NDRange(prevWidth, prevHeight));

		pPrevLayer = &_layers[l]._hiddenStatesFeedForward; // Or _hiddenStatesFeedBack ?
		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;

		pPrevLayerFeedForwardPrev = &_layers[l]._hiddenStatesFeedForwardPrev;
		pPrevLayerFeedBackPrev = &_layers[l]._hiddenStatesFeedBackPrev;
	}
}

void HTFE::stepEnd() {
	// ------------------------------------------------------------------------------
	// ---------------------------------- Step End ----------------------------------
	// ------------------------------------------------------------------------------

	for (int l = 0; l < _layers.size(); l++) {
		cl::Image2D temp2D;

		std::swap(_layers[l]._visibleReconstruction, _layers[l]._visibleReconstructionPrev);
		std::swap(_layers[l]._hiddenFeedBackActivations, _layers[l]._hiddenFeedBackActivationsPrev);
		std::swap(_layers[l]._hiddenStatesFeedForward, _layers[l]._hiddenStatesFeedForwardPrev);

		temp2D = _layers[l]._hiddenStatesFeedBackPrevPrev;
		_layers[l]._hiddenStatesFeedBackPrevPrev = _layers[l]._hiddenStatesFeedBackPrev;
		_layers[l]._hiddenStatesFeedBackPrev = _layers[l]._hiddenStatesFeedBack;
		_layers[l]._hiddenStatesFeedBack = temp2D;

		std::swap(_layers[l]._feedForwardWeights, _layers[l]._feedForwardWeightsPrev);
		std::swap(_layers[l]._reconstructionWeights, _layers[l]._reconstructionWeightsPrev);
		std::swap(_layers[l]._predictiveWeights, _layers[l]._predictiveWeightsPrev);
		std::swap(_layers[l]._lateralWeights, _layers[l]._lateralWeightsPrev);
		std::swap(_layers[l]._feedBackWeights, _layers[l]._feedBackWeightsPrev);
	}

	std::swap(_inputImage, _inputImagePrev);
}

void HTFE::clearMemory(sys::ComputeSystem &cs) {
	// ------------------------------------------------------------------------------
	// -------------------------------- Clear Memory --------------------------------
	// ------------------------------------------------------------------------------

	cl_uint4 clear = { 0, 0, 0, 0 };

	for (int l = 0; l < _layers.size(); l++) {
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _layerDescs[l]._width;
		region[1] = _layerDescs[l]._height;
		region[2] = 1;

		cs.getQueue().enqueueFillImage(_layers[l]._hiddenStatesFeedBackPrevPrev, clear, origin, region);
		cs.getQueue().enqueueFillImage(_layers[l]._hiddenStatesFeedBackPrev, clear, origin, region);
		cs.getQueue().enqueueFillImage(_layers[l]._hiddenStatesFeedBack, clear, origin, region);
	}
}

void HTFE::gaussianBlur(sys::ComputeSystem &cs, cl::Image2D &source, cl::Image2D &ping, cl::Image2D &pong, int imageSizeX, int imageSizeY, int passes, float kernelWidth) {
	Float2 imageSizeInv;
	imageSizeInv._x = 1.0f / imageSizeX;
	imageSizeInv._y = 1.0f / imageSizeY;

	// Blur source to ping
	_gaussianBlurXKernel.setArg(0, source);
	_gaussianBlurXKernel.setArg(1, ping);
	_gaussianBlurXKernel.setArg(2, imageSizeInv);
	_gaussianBlurXKernel.setArg(3, kernelWidth * imageSizeInv._x);

	cs.getQueue().enqueueNDRangeKernel(_gaussianBlurXKernel, cl::NullRange, cl::NDRange(imageSizeX, imageSizeY));

	for (int p = 0; p < passes - 1; p++) {
		_gaussianBlurYKernel.setArg(0, ping);
		_gaussianBlurYKernel.setArg(1, pong);
		_gaussianBlurYKernel.setArg(2, imageSizeInv);
		_gaussianBlurYKernel.setArg(3, kernelWidth * imageSizeInv._y);

		cs.getQueue().enqueueNDRangeKernel(_gaussianBlurYKernel, cl::NullRange, cl::NDRange(imageSizeX, imageSizeY));

		_gaussianBlurXKernel.setArg(0, pong);
		_gaussianBlurXKernel.setArg(1, ping);
		_gaussianBlurXKernel.setArg(2, imageSizeInv);
		_gaussianBlurXKernel.setArg(3, kernelWidth * imageSizeInv._x);

		cs.getQueue().enqueueNDRangeKernel(_gaussianBlurXKernel, cl::NullRange, cl::NDRange(imageSizeX, imageSizeY));
	}

	_gaussianBlurYKernel.setArg(0, ping);
	_gaussianBlurYKernel.setArg(1, pong);
	_gaussianBlurYKernel.setArg(2, imageSizeInv);
	_gaussianBlurYKernel.setArg(3, kernelWidth * imageSizeInv._y);

	cs.getQueue().enqueueNDRangeKernel(_gaussianBlurYKernel, cl::NullRange, cl::NDRange(imageSizeX, imageSizeY));
}