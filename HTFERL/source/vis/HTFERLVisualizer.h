#pragma once

#include <SFML/Graphics.hpp>
#include <htferl/HTFERL.h>

namespace vis {
	class HTFERLVisualizer {
	private:
		sf::RenderTexture _rt;
	public:
		void create(unsigned int width);

		void update(sf::RenderTexture &target, const sf::Vector2f &position, const sf::Vector2f &scale, sys::ComputeSystem &cs, const htferl::HTFERL &htmrl, std::mt19937 &generator);
	};
}