constant sampler_t normalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;
	
constant sampler_t normalizedClampedToEdgeNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;
	
constant sampler_t unnormalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;
	
constant sampler_t defaultNormalizedSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;
	
constant sampler_t defaultUnnormalizedSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;
	
float randFloat(uint2* state) {
    const float invMaxInt = 1.0f / 4294967296.0f;
    uint x = (*state).x * 17 + (*state).y * 13123;
    (*state).x = (x << 13) ^ x;
    (*state).y ^= (x << 7);

    uint tmp = x * (x * x * 15731 + 74323) + 871483;

    return convert_float(tmp) * invMaxInt;
}

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}
	
float boostFunction(float trace, float threshold) {
	return fmin(1.0f, fmax(0.0f, threshold - trace) / threshold);
}

void kernel initializeLayerHidden(write_only image2d_t hiddenActivations,
	write_only image2d_t hiddenStates,
	write_only image3d_t feedForwardWeights,
	write_only image2d_t hiddenBiases,
	write_only image2d_t hiddenVisibleBiases,
	write_only image3d_t lateralWeights,
	int feedForwardSize, int lateralSize,
	uint2 seed, float sparsity, float minWeight, float maxWeight)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 - 12, get_global_id(1) * 16 + 23) * 36;

	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	write_imagef(hiddenActivations, hiddenPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
	write_imagef(hiddenStates, hiddenPosition, (float4)(0.0f, sparsity, 0.0f, 0.0f));
	
	float hiddenBias = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;
	float hiddenVisibleBias = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;
	
	write_imagef(hiddenBiases, hiddenPosition, (float4)(hiddenBias, 0.0f, 0.0f, 0.0f));
	write_imagef(hiddenVisibleBiases, hiddenPosition, (float4)(hiddenVisibleBias, 0.0f, 0.0f, 0.0f));
	
	for (int wi = 0; wi < feedForwardSize; wi++) {
		int4 weightPosition = (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0);
	
		float feedForwardWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(feedForwardWeights, weightPosition, (float4)(feedForwardWeight, 0.0f, 0.0f, 0.0f));
	}
	
	for (int wi = 0; wi < lateralSize; wi++) {
		int4 weightPosition = (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0);
	
		float lateralWeight = randFloat(&seedValue) * (maxWeight - minWeight) + minWeight;

		write_imagef(lateralWeights, weightPosition, (float4)(lateralWeight, 0.0f, 0.0f, 0.0f));
	}
}

void kernel initializeLayerVisible(write_only image2d_t visibleBiases,
	uint2 seed, float minWeight, float maxWeight)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 - 12, get_global_id(1) * 16 + 23) * 36;

	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));

	write_imagef(visibleBiases, visiblePosition, (float4)(randFloat(&seedValue), 0.0f, 0.0f, 0.0f));
}

void kernel layerHiddenActivate(read_only image2d_t inputs, read_only image2d_t hiddenStatesPrev, read_only image3d_t feedForwardWeights, read_only image3d_t lateralWeights, read_only image2d_t hiddenBiases, write_only image2d_t hiddenActivations,
	int2 layerSize, float2 layerSizeMinusOneInv, int2 inputSize, int2 inputSizeMinusOne, int receptiveFieldRadius, int lateralConnectionRadius)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 inputCenterPositionNormalized = (float2)(hiddenPosition.x * layerSizeMinusOneInv.x, hiddenPosition.y * layerSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);

	float sum = 0.0f;

	int wi = 0;

	for (int dx = -receptiveFieldRadius; dx <= receptiveFieldRadius; dx++)
	for (int dy = -receptiveFieldRadius; dy <= receptiveFieldRadius; dy++) {
		int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float input = read_imagef(inputs, inputPosition).x;
	
			float weight = read_imagef(feedForwardWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;
				
			sum += weight * input;
		}
		
		wi++;
	}
	
	for (int dx = -lateralConnectionRadius; dx <= lateralConnectionRadius; dx++)
	for (int dy = -lateralConnectionRadius; dy <= lateralConnectionRadius; dy++) {
		int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);
		
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float state = read_imagef(hiddenStatesPrev, layerPosition).x;
	
			float weight = read_imagef(lateralWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;
				
			sum += weight * state;
		}
		
		wi++;
	}
	
	// Bias
	float bias = read_imagef(hiddenBiases, hiddenPosition).x;
	
	sum += bias;
	
	write_imagef(hiddenActivations, hiddenPosition, (float4)(sigmoid(sum), 0.0f, 0.0f, 0.0f));
}

void kernel layerHiddenInhibit(read_only image2d_t hiddenActivations, read_only image2d_t hiddenStatesPrev, write_only image2d_t hiddenStates,
	int2 layerSize, int inhibitionRadius, float localActivity, float dutyCycleDecay)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float thisActivation = read_imagef(hiddenActivations, hiddenPosition).x;
	
	float numHigher = 0.0f;
	
	for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
	for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
		int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);
		
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float activation = read_imagef(hiddenActivations, layerPosition).x;
	
			numHigher += activation > thisActivation ? 1.0f : 0.0f;
		}
	}
	
	float prevDutyCycle = read_imagef(hiddenStatesPrev, hiddenPosition).y;
	
	float newState = numHigher < localActivity ? 1.0f : 0.0f;
	
	float newDutyCycle = (1.0f - dutyCycleDecay) * prevDutyCycle + dutyCycleDecay * newState;
	
	write_imagef(hiddenStates, hiddenPosition, (float4)(newState, newDutyCycle, 0.0f, 0.0f));
}

void kernel layerVisibleReconstruct(read_only image2d_t hiddenStates, read_only image3d_t feedForwardWeights, read_only image2d_t visibleBiases, write_only image2d_t visibleReconstruction,
	int2 reverseReceptiveFieldRadius, int2 layerReceptiveFieldRadius, int2 inputSizeMinusOne, float2 inputSizeMinusOneInv, int2 layerSize, int2 layerSizeMinusOne, float2 layerSizeMinusOneInv)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	float2 layerPositionNormalized = (float2)(visiblePosition.x * inputSizeMinusOneInv.x, visiblePosition.y * inputSizeMinusOneInv.y);
	float2 layerPositionCenter = (float2)(layerPositionNormalized.x * layerSizeMinusOne.x, layerPositionNormalized.y * layerSizeMinusOne.y);
	
	float sum = 0.0f;

	for (int dx = -reverseReceptiveFieldRadius.x; dx <= reverseReceptiveFieldRadius.x; dx++)
	for (int dy = -reverseReceptiveFieldRadius.y; dy <= reverseReceptiveFieldRadius.y; dy++) {
		int2 layerPosition = (int2)(layerPositionCenter.x + dx, layerPositionCenter.y + dy);
		
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			// Next layer node's receptive field
			int2 fieldCenter = (int2)(layerPosition.x * layerSizeMinusOneInv.x * inputSizeMinusOne.x, layerPosition.y * layerSizeMinusOneInv.y * inputSizeMinusOne.y);

			int2 fieldLowerBounds = fieldCenter - layerReceptiveFieldRadius;
			int2 fieldUpperBounds = fieldCenter + layerReceptiveFieldRadius;
		
			// Check for containment
			if (visiblePosition.x >= fieldLowerBounds.x && visiblePosition.x <= fieldUpperBounds.x && visiblePosition.y >= fieldLowerBounds.y && visiblePosition.y <= fieldUpperBounds.y) {	
				int rdx = visiblePosition.x - fieldCenter.x;
				int rdy = visiblePosition.y - fieldCenter.y;
				
				float source = read_imagef(hiddenStates, layerPosition).x;

				int weightIndex = (layerReceptiveFieldRadius.y + rdy) + (layerReceptiveFieldRadius.x + rdx) * (layerReceptiveFieldRadius.y * 2 + 1);

				float weight = read_imagef(feedForwardWeights, (int4)(layerPosition.x, layerPosition.y, weightIndex, 0)).x;
				
				sum += source * weight;
			}
		}
	}

	float bias = read_imagef(visibleBiases, visiblePosition).x;
				
	sum += bias;
	
	write_imagef(visibleReconstruction, visiblePosition, (float4)(sum, 0.0f, 0.0f, 0.0f));
}

void kernel layerHiddenReconstruct(read_only image2d_t hiddenStates, read_only image3d_t lateralWeights, read_only image2d_t hiddenVisibleBiases, write_only image2d_t hiddenReconstruction,
	int lateralConnectionRadius, int inhibitionRadius, int2 layerSize)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float sum = 0.0f;
	
	for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
	for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
		int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);
		
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float source = read_imagef(hiddenStates, layerPosition).x;
			
			float weightIndex = (inhibitionRadius + dy) + (inhibitionRadius + dx) * (inhibitionRadius * 2 + 1);
			
			float weight = read_imagef(lateralWeights, (int4)(hiddenPosition.x, hiddenPosition.y, weightIndex, 0)).x;
			
			sum += source * weight;
		}
	}
	
	float bias = read_imagef(hiddenVisibleBiases, hiddenPosition).x;
	
	sum += bias;
	
	write_imagef(hiddenReconstruction, hiddenPosition, (float4)(sum, 0.0f, 0.0f, 0.0f));
}

void kernel layerWeightUpdate(read_only image2d_t visibleReconstruction, read_only image2d_t hiddenReconstruction, read_only image2d_t inputs, read_only image2d_t hiddenActivations, read_only image2d_t hiddenStates,
	read_only image2d_t hiddenStatesPrev, read_only image3d_t feedForwardWeightsPrev, read_only image3d_t lateralWeightsPrev, read_only image2d_t hiddenBiasesPrev,
	write_only image3d_t feedForwardWeights, write_only image3d_t lateralWeights, write_only image2d_t hiddenBiases,
	int2 layerSize, float2 layerSizeMinusOneInv, int2 inputSize, int2 inputSizeMinusOne, int receptiveFieldRadius, int lateralConnectionRadius, float sparsity, float alpha, float beta, float traceDecay) 
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 inputCenterPositionNormalized = (float2)(hiddenPosition.x * layerSizeMinusOneInv.x, hiddenPosition.y * layerSizeMinusOneInv.y);
	float2 inputCenterPosition = (float2)(inputCenterPositionNormalized.x * inputSizeMinusOne.x, inputCenterPositionNormalized.y * inputSizeMinusOne.y);

	float2 thisHiddenState = read_imagef(hiddenStates, hiddenPosition).xy;
	float thisHiddenActivation = read_imagef(hiddenActivations, hiddenPosition).x;

	// --------------------------------- Collect Error -------------------------------------
	
	float sum = 0.0f;
	
	int wi = 0;
	
	for (int dx = -receptiveFieldRadius; dx <= receptiveFieldRadius; dx++)
	for (int dy = -receptiveFieldRadius; dy <= receptiveFieldRadius; dy++) {
		int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float recon = read_imagef(visibleReconstruction, inputPosition).x;
			float input = read_imagef(inputs, inputPosition).x;
			
			float2 prevWeight = read_imagef(feedForwardWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;
			
			// For bias update
			sum += (input - recon) * prevWeight.x;
		}
		
		wi++;
	}
	
	wi = 0;
	
	for (int dx = -lateralConnectionRadius; dx <= lateralConnectionRadius; dx++)
	for (int dy = -lateralConnectionRadius; dy <= lateralConnectionRadius; dy++) {
		int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);
		
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float recon = read_imagef(hiddenReconstruction, layerPosition).x;
			float input = read_imagef(hiddenStatesPrev, layerPosition).x;
			
			float2 prevWeight = read_imagef(lateralWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

			sum += (input - recon) * prevWeight.x;
		}
		
		wi++;
	}
	
	float error = sum * thisHiddenActivation * (1.0f - thisHiddenActivation);
	
	// --------------------------------- Update on Error ---------------------------------
	
	wi = 0;

	for (int dx = -receptiveFieldRadius; dx <= receptiveFieldRadius; dx++)
	for (int dy = -receptiveFieldRadius; dy <= receptiveFieldRadius; dy++) {
		int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);
		
		if (inputPosition.x >= 0 && inputPosition.x < inputSize.x && inputPosition.y >= 0 && inputPosition.y < inputSize.y) {
			float recon = read_imagef(visibleReconstruction, inputPosition).x;
			float input = read_imagef(inputs, inputPosition).x;
			
			float eligibility = 0.5f * (error * input + (input - recon) * thisHiddenState.x);
	
			float2 prevWeight = read_imagef(feedForwardWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;
				
			float2 newWeight = (float2)(prevWeight.x + alpha * prevWeight.y, (1.0f - traceDecay) * prevWeight.y + beta * eligibility);
			
			write_imagef(feedForwardWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(newWeight.x, newWeight.y, 0.0f, 0.0f));
		}
		
		wi++;
	}
	
	wi = 0;

	for (int dx = -lateralConnectionRadius; dx <= lateralConnectionRadius; dx++)
	for (int dy = -lateralConnectionRadius; dy <= lateralConnectionRadius; dy++) {
		int2 layerPosition = (int2)(hiddenPosition.x + dx, hiddenPosition.y + dy);
		
		if (layerPosition.x >= 0 && layerPosition.x < layerSize.x && layerPosition.y >= 0 && layerPosition.y < layerSize.y) {
			float recon = read_imagef(hiddenReconstruction, layerPosition).x;
			float input = read_imagef(hiddenStatesPrev, layerPosition).x;
			
			float eligibility = 0.5f * (error * input + (input - recon) * thisHiddenState.x);
	
			float2 prevWeight = read_imagef(lateralWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;
				
			float2 newWeight = (float2)(prevWeight.x + alpha * prevWeight.y, (1.0f - traceDecay) * prevWeight.y + beta * eligibility);
			
			write_imagef(lateralWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(newWeight.x, newWeight.y, 0.0f, 0.0f));
		}
		
		wi++;
	}
	
	float eligibility = error;
	
	float2 prevBias = read_imagef(hiddenBiasesPrev, hiddenPosition).xy;
		
	float2 newBias = (float2)(prevBias.x + alpha * prevBias.y, (1.0f - traceDecay) * prevBias.y + beta * eligibility);
	
	write_imagef(hiddenBiases, hiddenPosition, (float4)(newBias.x, newBias.y, 0.0f, 0.0f));
}

void kernel layerVisibleBiasUpdate(read_only image2d_t inputs, read_only image2d_t visibleReconstruction, read_only image2d_t visibleBiasesPrev, write_only image2d_t visibleBiases,
	float alpha, float beta, float traceDecay)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	
	float recon = read_imagef(visibleReconstruction, visiblePosition).x;
	float input = read_imagef(inputs, visiblePosition).x;
	
	float eligibility = input - recon;

	float2 prevWeight = read_imagef(visibleBiasesPrev, visiblePosition).xy;
		
	float2 newWeight = (float2)(prevWeight.x + alpha * prevWeight.y, (1.0f - traceDecay) * prevWeight.y + beta * eligibility);
	
	write_imagef(visibleBiases, visiblePosition, (float4)(newWeight.x, newWeight.y, 0.0f, 0.0f));
}

void kernel layerHiddenVisibleBiasUpdate(read_only image2d_t hiddenStatesPrev, read_only image2d_t hiddenReconstruction, read_only image2d_t hiddenVisibleBiasesPrev, write_only image2d_t hiddenVisibleBiases,
	float alpha, float beta, float traceDecay)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float recon = read_imagef(hiddenReconstruction, hiddenPosition).x;
	float input = read_imagef(hiddenStatesPrev, hiddenPosition).x;
	
	float eligibility = input - recon;

	float2 prevWeight = read_imagef(hiddenVisibleBiasesPrev, hiddenPosition).xy;
		
	float2 newWeight = (float2)(prevWeight.x + alpha * prevWeight.y, (1.0f - traceDecay) * prevWeight.y + beta * eligibility);
	
	write_imagef(hiddenVisibleBiases, hiddenPosition, (float4)(newWeight.x, newWeight.y, 0.0f, 0.0f));
}