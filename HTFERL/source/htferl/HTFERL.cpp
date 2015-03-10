#include "HTFERL.h"

#include <iostream>

using namespace htferl;

void HTFERL::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, std::mt19937 &generator) {
	struct Uint2 {
		unsigned int _x, _y;
	};

	std::uniform_int_distribution<int> seedDist(0, 99999);

	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_layerDescs = layerDescs;
	
	_layers.resize(_layerDescs.size());

	_inputTypes = inputTypes;

	std::uniform_real_distribution<float> weightDist(minInitWeight, maxInitWeight);
	std::uniform_real_distribution<float> actionDist(0.0f, 1.0f);

	_prevMax = 0.0f;
	_prevValue = 0.0f;

	cl::Kernel initializeLayerHiddenKernel = cl::Kernel(program.getProgram(), "initializeLayerHidden");
	cl::Kernel initializeLayerVisibleKernel = cl::Kernel(program.getProgram(), "initializeLayerVisible");

	_input.clear();
	_input.resize(_inputWidth * _inputHeight);

	_prevMaxInput.clear();
	_prevMaxInput.resize(_inputWidth * _inputHeight);

	// Initialize action portions randomly
	for (int i = 0; i < _input.size(); i++)
		if (_inputTypes[i] == _action) {
			float value = actionDist(generator);

			_input[i] = value;
			_prevMaxInput[i] = value;
		}
		else
			_input[i] = _prevMaxInput[i] = 0.0f;

	_inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);
	_inputImagePrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	_learnImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	int prevWidth = _inputWidth;
	int prevHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		int numFeedForwardWeights = std::pow(_layerDescs[l]._receptiveFieldRadius * 2 + 1, 2);
		int numReconstructionWeights = std::pow(_layerDescs[l]._reconstructionRadius * 2 + 1, 2);
		int numLateralWeights = std::pow(_layerDescs[l]._lateralConnectionRadius * 2 + 1, 2);
		int numFeedBackWeights = std::pow(_layerDescs[l]._feedBackConnectionRadius * 2 + 1, 2);

		_layers[l]._hiddenFeedForwardActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		
		_layers[l]._hiddenFeedBackActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._hiddenFeedBackActivationsPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._hiddenStatesFeedForward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._hiddenStatesFeedForwardPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._hiddenStatesFeedBack = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._hiddenStatesFeedBackPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._hiddenStatesFeedBackPrevPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._feedForwardWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numFeedForwardWeights);
		_layers[l]._feedForwardWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numFeedForwardWeights);

		_layers[l]._reconstructionWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), prevWidth, prevHeight, numReconstructionWeights);
		_layers[l]._reconstructionWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), prevWidth, prevHeight, numReconstructionWeights);

		_layers[l]._visibleBiases = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), prevWidth, prevHeight);
		_layers[l]._visibleBiasesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), prevWidth, prevHeight);

		_layers[l]._hiddenBiases = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);
		_layers[l]._hiddenBiasesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height);

		_layers[l]._lateralWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numLateralWeights);
		_layers[l]._lateralWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numLateralWeights);

		_layers[l]._feedBackWeights = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numFeedBackWeights);
		_layers[l]._feedBackWeightsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _layerDescs[l]._width, _layerDescs[l]._height, numFeedBackWeights);

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
		initializeLayerHiddenKernel.setArg(index++, _layers[l]._hiddenBiases);
		initializeLayerHiddenKernel.setArg(index++, _layers[l]._lateralWeights);
		initializeLayerHiddenKernel.setArg(index++, numFeedForwardWeights);
		initializeLayerHiddenKernel.setArg(index++, numLateralWeights);
		initializeLayerHiddenKernel.setArg(index++, initSeedHidden);
		initializeLayerHiddenKernel.setArg(index++, _layerDescs[l]._sparsity);
		initializeLayerHiddenKernel.setArg(index++, minInitWeight);
		initializeLayerHiddenKernel.setArg(index++, maxInitWeight);

		cs.getQueue().enqueueNDRangeKernel(initializeLayerHiddenKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		Uint2 initSeedVisible;
		initSeedVisible._x = seedDist(generator);
		initSeedVisible._y = seedDist(generator);

		index = 0;

		initializeLayerVisibleKernel.setArg(index++, _layers[l]._visibleBiases);
		initializeLayerVisibleKernel.setArg(index++, _layers[l]._visibleReconstruction);
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
			region[0] = prevWidth;
			region[1] = prevHeight;
			region[2] = 1;

			cs.getQueue().enqueueCopyImage(_layers[l]._visibleBiases, _layers[l]._visibleBiasesPrev, origin, origin, region);
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

			cs.getQueue().enqueueCopyImage(_layers[l]._hiddenBiases, _layers[l]._hiddenBiasesPrev, origin, origin, region);
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
	_layerVisibleReconstructKernel = cl::Kernel(program.getProgram(), "layerVisibleReconstruct");
	_layerHiddenWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerHiddenWeightUpdate");
	_layerHiddenWeightUpdateLastKernel = cl::Kernel(program.getProgram(), "layerHiddenWeightUpdateLast");
	_layerVisibleWeightUpdateKernel = cl::Kernel(program.getProgram(), "layerVisibleWeightUpdate");
}

void HTFERL::step(sys::ComputeSystem &cs, float reward, float minLearn, float alpha, float gamma, float breakChance, float perturbationStdDev, std::mt19937 &generator) {
	struct Float2 {
		float _x, _y;
	};

	struct Int2 {
		int _x, _y;
	};
	
	// Step begin
	for (int l = 0; l < _layers.size(); l++) {
		cl::Image2D temp2D;

		std::swap(_layers[l]._visibleReconstruction, _layers[l]._visibleReconstructionPrev);
		std::swap(_layers[l]._hiddenStatesFeedForward, _layers[l]._hiddenStatesFeedForwardPrev);
		//std::swap(_layers[l]._hiddenStates, _layers[l]._hiddenStatesPrev);
		temp2D = _layers[l]._hiddenStatesFeedBackPrevPrev;
		_layers[l]._hiddenStatesFeedBackPrevPrev = _layers[l]._hiddenStatesFeedBackPrev;
		_layers[l]._hiddenStatesFeedBackPrev = _layers[l]._hiddenStatesFeedBack;
		_layers[l]._hiddenStatesFeedBack = temp2D;

		std::swap(_layers[l]._feedForwardWeights, _layers[l]._feedForwardWeightsPrev);
		std::swap(_layers[l]._reconstructionWeights, _layers[l]._reconstructionWeightsPrev);
		std::swap(_layers[l]._visibleBiases, _layers[l]._visibleBiasesPrev);
		std::swap(_layers[l]._hiddenBiases, _layers[l]._hiddenBiasesPrev);
		std::swap(_layers[l]._lateralWeights, _layers[l]._lateralWeightsPrev);
		std::swap(_layers[l]._feedBackWeights, _layers[l]._feedBackWeightsPrev);
	}
		
	std::swap(_inputImage, _inputImagePrev);

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

		Int2 receptiveFieldRadius;
		receptiveFieldRadius._x = _layerDescs[l]._receptiveFieldRadius;
		receptiveFieldRadius._y = _layerDescs[l]._receptiveFieldRadius;

		// -------------------------------- Activate --------------------------------

		int index = 0;

		_layerHiddenFeedForwardActivateKernel.setArg(index++, *pPrevLayer);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrev);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, _layers[l]._feedForwardWeightsPrev);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, _layers[l]._lateralWeightsPrev);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, _layers[l]._hiddenBiasesPrev);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, _layers[l]._hiddenFeedForwardActivations);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, layerSize);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, layerSizeMinusOneInv);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, inputSize);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, inputSizeMinusOne);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, _layerDescs[l]._receptiveFieldRadius);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, _layerDescs[l]._lateralConnectionRadius);
		_layerHiddenFeedForwardActivateKernel.setArg(index++, _layerDescs[l]._lateralScalar);

		cs.getQueue().enqueueNDRangeKernel(_layerHiddenFeedForwardActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		// ---------------------------------- Inhibit ---------------------------------

		index = 0;

		_layerHiddenInhibitKernel.setArg(index++, _layers[l]._hiddenFeedForwardActivations);
		_layerHiddenInhibitKernel.setArg(index++, _layers[l]._hiddenStatesFeedForwardPrev);
		_layerHiddenInhibitKernel.setArg(index++, _layers[l]._hiddenStatesFeedForward);
		_layerHiddenInhibitKernel.setArg(index++, layerSize);
		_layerHiddenInhibitKernel.setArg(index++, _layerDescs[l]._inhibitionRadius);
		_layerHiddenInhibitKernel.setArg(index++, localActivity);
		_layerHiddenInhibitKernel.setArg(index++, _layerDescs[l]._dutyCycleDecay);
		_layerHiddenInhibitKernel.setArg(index++, _layerDescs[l]._sdrDecay);

		cs.getQueue().enqueueNDRangeKernel(_layerHiddenInhibitKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

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

		Int2 receptiveFieldRadius;
		receptiveFieldRadius._x = _layerDescs[l]._receptiveFieldRadius;
		receptiveFieldRadius._y = _layerDescs[l]._receptiveFieldRadius;

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
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layers[l]._hiddenFeedForwardActivations);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layers[l + 1]._hiddenFeedBackActivations);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layers[l]._feedBackWeightsPrev);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layers[l]._hiddenFeedBackActivations);
			_layerHiddenFeedBackActivateKernel.setArg(index++, layerSize);
			_layerHiddenFeedBackActivateKernel.setArg(index++, layerSizeMinusOneInv);
			_layerHiddenFeedBackActivateKernel.setArg(index++, nextSize);
			_layerHiddenFeedBackActivateKernel.setArg(index++, nextSizeMinusOne);
			_layerHiddenFeedBackActivateKernel.setArg(index++, _layerDescs[l]._feedBackConnectionRadius);

			cs.getQueue().enqueueNDRangeKernel(_layerHiddenFeedBackActivateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}

		// ---------------------------------- Inhibit ---------------------------------

		index = 0;

		_layerHiddenInhibitKernel.setArg(index++, _layers[l]._hiddenFeedBackActivations);
		_layerHiddenInhibitKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrev);
		_layerHiddenInhibitKernel.setArg(index++, _layers[l]._hiddenStatesFeedBack);
		_layerHiddenInhibitKernel.setArg(index++, layerSize);
		_layerHiddenInhibitKernel.setArg(index++, _layerDescs[l]._inhibitionRadius);
		_layerHiddenInhibitKernel.setArg(index++, localActivity);
		_layerHiddenInhibitKernel.setArg(index++, _layerDescs[l]._dutyCycleDecay);
		_layerHiddenInhibitKernel.setArg(index++, _layerDescs[l]._sdrDecay);

		cs.getQueue().enqueueNDRangeKernel(_layerHiddenInhibitKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));

		// --------------------- Make Predictions (Reconstruction) ---------------------

		index = 0;

		_layerVisibleReconstructKernel.setArg(index++, _layers[l]._hiddenStatesFeedBack);
		_layerVisibleReconstructKernel.setArg(index++, _layers[l]._reconstructionWeightsPrev);
		_layerVisibleReconstructKernel.setArg(index++, _layers[l]._visibleBiasesPrev);
		_layerVisibleReconstructKernel.setArg(index++, _layers[l]._visibleReconstruction);
		_layerVisibleReconstructKernel.setArg(index++, _layerDescs[l]._reconstructionRadius);
		_layerVisibleReconstructKernel.setArg(index++, inputSizeMinusOne);
		_layerVisibleReconstructKernel.setArg(index++, inputSizeMinusOneInv);
		_layerVisibleReconstructKernel.setArg(index++, layerSize);
		_layerVisibleReconstructKernel.setArg(index++, layerSizeMinusOne);
		_layerVisibleReconstructKernel.setArg(index++, layerSizeMinusOneInv);

		cs.getQueue().enqueueNDRangeKernel(_layerVisibleReconstructKernel, cl::NullRange, cl::NDRange(prevWidth, prevHeight));
	}

	// ------------------------------------------------------------------------------
	// ----------------------------- Q Value Updates  -------------------------------
	// ------------------------------------------------------------------------------

	// Exploratory action
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, perturbationStdDev);

	std::vector<float> output(_input.size());

	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueReadImage(_layers.front()._visibleReconstruction, CL_TRUE, origin, region, 0, 0, output.data());
	}

	float nextQ = 0.0f;
	float qDiv = 0.0f;

	std::vector<float> newInput(_input.size());

	for (int i = 0; i < _input.size(); i++)
		if (_inputTypes[i] == _action) {
			if (dist01(generator) < breakChance)
				newInput[i] = dist01(generator);
			else
				newInput[i] = std::min<float>(1.0f, std::max<float>(0.0f, std::min<float>(1.0f, std::max<float>(0.0f, output[i])) + pertDist(generator)));
		}
		else if (_inputTypes[i] == _q) {
			newInput[i] = output[i];

			nextQ += output[i];

			qDiv++;
		}

	nextQ /= qDiv;

	float tdError = reward + gamma * nextQ - _prevValue;

	float newQ = _prevValue + alpha * tdError;

	_prevValue = nextQ;

	std::vector<float> learnData(_input.size());

	if (tdError > 0.0f) {
		for (int i = 0; i < learnData.size(); i++)
			if (_inputTypes[i] == _action) {
				learnData[i] = _input[i];
			}
			else if (_inputTypes[i] == _q) {
				_input[i] = nextQ;
				learnData[i] = newQ;
			}
			else
				learnData[i] = _input[i];
	}
	else {
		for (int i = 0; i < learnData.size(); i++)
			if (_inputTypes[i] == _action) {
				learnData[i] = _prevMaxInput[i];
			}
			else if (_inputTypes[i] == _q) {
				_input[i] = nextQ;
				learnData[i] = newQ;
			}
			else
				learnData[i] = _input[i];
	}

	_input = newInput;

	std::cout << nextQ << " " << tdError << std::endl;

	float learnAction = 1.0f;

	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueWriteImage(_learnImage, CL_TRUE, origin, region, 0, 0, learnData.data());
	}

	// ------------------------------------------------------------------------------
	// ---------------------- Weight Update and Predictions  ------------------------
	// ------------------------------------------------------------------------------

	pPrevLayer = &_learnImage;
	prevWidth = _inputWidth;
	prevHeight = _inputHeight;

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

		Int2 receptiveFieldRadius;
		receptiveFieldRadius._x = _layerDescs[l]._receptiveFieldRadius;
		receptiveFieldRadius._y = _layerDescs[l]._receptiveFieldRadius;

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

		Float2 alphas;
		alphas._x = _layerDescs[l]._feedForwardAlpha * learnAction;
		alphas._y = _layerDescs[l]._lateralAlpha * learnAction;

		Float2 betas;
		betas._x = _layerDescs[l]._feedForwardBeta;
		betas._y = _layerDescs[l]._lateralBeta;

		int index = 0;

		if (l == _layers.size() - 1) {
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._visibleReconstructionPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, *pPrevLayer);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, *pPrevLayerFeedForwardPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._hiddenFeedBackActivationsPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrevPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._reconstructionWeightsPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._feedForwardWeightsPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._lateralWeightsPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._hiddenBiasesPrev);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._feedForwardWeights);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._lateralWeights);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layers[l]._hiddenBiases);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, layerSize);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, layerSizeMinusOne);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, layerSizeMinusOneInv);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, inputSize);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, inputSizeMinusOne);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, inputSizeMinusOneInv);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layerDescs[l]._receptiveFieldRadius);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layerDescs[l]._lateralConnectionRadius);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layerDescs[l]._reconstructionRadius);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layerDescs[l]._sparsity);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, alphas);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, betas);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layerDescs[l]._gamma);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layerDescs[l]._temperature);
			_layerHiddenWeightUpdateLastKernel.setArg(index++, _layerDescs[l]._traceDecay);

			cs.getQueue().enqueueNDRangeKernel(_layerHiddenWeightUpdateLastKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}
		else {
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._visibleReconstructionPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, *pPrevLayer);
			_layerHiddenWeightUpdateKernel.setArg(index++, *pPrevLayerFeedForwardPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._hiddenFeedBackActivationsPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrevPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l + 1]._hiddenStatesFeedBackPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._reconstructionWeightsPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._feedForwardWeightsPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._lateralWeightsPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._hiddenBiasesPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._feedBackWeightsPrev);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._feedForwardWeights);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._lateralWeights);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._hiddenBiases);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layers[l]._feedBackWeights);
			_layerHiddenWeightUpdateKernel.setArg(index++, layerSize);
			_layerHiddenWeightUpdateKernel.setArg(index++, layerSizeMinusOne);
			_layerHiddenWeightUpdateKernel.setArg(index++, layerSizeMinusOneInv);
			_layerHiddenWeightUpdateKernel.setArg(index++, inputSize);
			_layerHiddenWeightUpdateKernel.setArg(index++, inputSizeMinusOne);
			_layerHiddenWeightUpdateKernel.setArg(index++, inputSizeMinusOneInv);
			_layerHiddenWeightUpdateKernel.setArg(index++, nextSize);
			_layerHiddenWeightUpdateKernel.setArg(index++, nextSizeMinusOne);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._receptiveFieldRadius);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._lateralConnectionRadius);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._feedBackConnectionRadius);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._reconstructionRadius);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._sparsity);
			_layerHiddenWeightUpdateKernel.setArg(index++, alphas);
			_layerHiddenWeightUpdateKernel.setArg(index++, betas);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._gamma);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._temperature);
			_layerHiddenWeightUpdateKernel.setArg(index++, _layerDescs[l]._traceDecay);

			cs.getQueue().enqueueNDRangeKernel(_layerHiddenWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._width, _layerDescs[l]._height));
		}

		index = 0;

		_layerVisibleWeightUpdateKernel.setArg(index++, _layers[l]._visibleReconstructionPrev);
		_layerVisibleWeightUpdateKernel.setArg(index++, *pPrevLayer);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layers[l]._hiddenStatesFeedBackPrev);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layers[l]._reconstructionWeightsPrev);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layers[l]._visibleBiasesPrev);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layers[l]._reconstructionWeights);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layers[l]._visibleBiases);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layerDescs[l]._reconstructionRadius);
		_layerVisibleWeightUpdateKernel.setArg(index++, inputSizeMinusOne);
		_layerVisibleWeightUpdateKernel.setArg(index++, inputSizeMinusOneInv);
		_layerVisibleWeightUpdateKernel.setArg(index++, layerSize);
		_layerVisibleWeightUpdateKernel.setArg(index++, layerSizeMinusOne);
		_layerVisibleWeightUpdateKernel.setArg(index++, layerSizeMinusOneInv);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layerDescs[l]._feedForwardAlpha * learnAction);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layerDescs[l]._feedForwardBeta);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layerDescs[l]._temperature);
		_layerVisibleWeightUpdateKernel.setArg(index++, _layerDescs[l]._traceDecay);

		cs.getQueue().enqueueNDRangeKernel(_layerVisibleWeightUpdateKernel, cl::NullRange, cl::NDRange(prevWidth, prevHeight));

		pPrevLayer = &_layers[l]._hiddenStatesFeedBack; // Or _hiddenStatesFeedBack ?
		prevWidth = _layerDescs[l]._width;
		prevHeight = _layerDescs[l]._height;

		pPrevLayerFeedForwardPrev = &_layers[l]._hiddenStatesFeedForwardPrev;
		pPrevLayerFeedBackPrev = &_layers[l]._hiddenStatesFeedBackPrev;
	}

	_prevMaxInput = output;
}

void HTFERL::exportStateData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const {
	std::mt19937 generator(seed);
	
	int maxWidth = _inputWidth;
	int maxHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		maxWidth = std::max<int>(maxWidth, _layerDescs[l]._width);
		maxHeight = std::max<int>(maxHeight, _layerDescs[l]._height);
	}
	
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
		std::vector<float> state(_inputWidth * _inputHeight * 4);

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueReadImage(_layers.front()._hiddenStatesFeedBack, CL_TRUE, origin, region, 0, 0, &state[0]);

		sf::Color c;
		c.r = uniformDist(generator) * 255.0f;
		c.g = uniformDist(generator) * 255.0f;
		c.b = uniformDist(generator) * 255.0f;

		// Convert to colors
		std::shared_ptr<sf::Image> image = std::make_shared<sf::Image>();

		image->create(maxWidth, maxHeight, sf::Color::Transparent);

		for (int x = 0; x < _inputWidth; x++)
		for (int y = 0; y < _inputHeight; y++) {
			sf::Color color;

			color = c;

			color.a = std::min<float>(1.0f, std::max<float>(0.0f, state[4 * (x + y * _inputWidth)])) * (255.0f - 3.0f) + 3;

			image->setPixel(x - _inputWidth / 2 + maxWidth / 2, y - _inputHeight / 2 + maxHeight / 2, color);
		}

		images.push_back(image);
	}
	else {
		std::vector<float> state(_inputWidth * _inputHeight);

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = _inputWidth;
		region[1] = _inputHeight;
		region[2] = 1;

		cs.getQueue().enqueueReadImage(_layers.front()._visibleReconstruction, CL_TRUE, origin, region, 0, 0, &state[0]);

		sf::Color c;
		c.r = uniformDist(generator) * 255.0f;
		c.g = uniformDist(generator) * 255.0f;
		c.b = uniformDist(generator) * 255.0f;

		// Convert to colors
		std::shared_ptr<sf::Image> image = std::make_shared<sf::Image>();

		image->create(maxWidth, maxHeight, sf::Color::Transparent);

		for (int x = 0; x < _inputWidth; x++)
			for (int y = 0; y < _inputHeight; y++) {
			sf::Color color;

			color = c;

			color.a = std::min<float>(1.0f, std::max<float>(0.0f, state[(x + y * _inputWidth)])) * (255.0f - 3.0f) + 3;

			image->setPixel(x - _inputWidth / 2 + maxWidth / 2, y - _inputHeight / 2 + maxHeight / 2, color);
			}

		images.push_back(image);
	}
}