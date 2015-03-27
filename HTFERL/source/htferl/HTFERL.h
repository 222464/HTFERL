#pragma once

#include "../htfe/HTFE.h"
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

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		struct OutputConnection {
			float _weight;
			//float _trace;

			OutputConnection()
			//	: _trace(0.0f)
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

		struct ReplaySample {
			std::vector<float> _hiddenStates;

			std::vector<float> _action;
			std::vector<float> _maxAction;

			float _q;
			float _originalQ;
		};

		std::vector<float> _input;
		std::vector<InputType> _inputTypes;

		std::vector<ActionNode> _actionNodes;
		std::vector<QNode> _qNodes;

		float _prevValue;

		htfe::HTFE _htfe;

		std::list<ReplaySample> _replaySamples;

		std::vector<float> _actionPrev;
		std::vector<float> _maxActionPrev;
		std::vector<float> _hiddenStatesPrev;

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<htfe::LayerDesc> &layerDescs, const std::vector<InputType> &inputTypes, float minInitWeight, float maxInitWeight, std::mt19937 &generator);
	
		void step(sys::ComputeSystem &cs, float reward, float qAlpha, float qGamma, float breakChance, float perturbationStdDev, float alphaQ, float alphaAction, float qTraceDecay, float actionTraceDecay, float actionTraceBeta, float actionTraceTemperature, int replayChainSize, int replayCount, std::mt19937 &generator);

		int getInputWidth() const {
			return _htfe.getInputWidth();
		}

		int getInputHeight() const {
			return _htfe.getInputHeight();
		}

		int getNumActions() const {
			return _actionNodes.size();
		}

		int getNumQNodes() const {
			return _qNodes.size();
		}

		const std::vector<htfe::LayerDesc> &getLayerDescs() const {
			return _htfe.getLayerDescs();
		}

		void setInput(int i, float value) {
			_input[i] = value;
		}

		void setInput(int x, int y, float value) {
			setInput(x + y * _htfe.getInputWidth(), value);
		}

		float getOutput(int i) const {
			return _actionNodes[i]._output;
		}

		void exportStateData(sys::ComputeSystem &cs, std::vector<std::shared_ptr<sf::Image>> &images, unsigned long seed) const;
	};
}