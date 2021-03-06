#include <htferl/HTFERL.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <vis/HTFERLVisualizer.h>
#include <vis/Plot.h>

#include <time.h>
#include <iostream>
#include <functional>
#include <unordered_map>

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram program;

	program.loadFromFile("resources/htfe.cl", cs);

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	sf::RenderWindow window;

	window.create(sf::VideoMode(800, 600), "Pole Balancing");

	//window.setVerticalSyncEnabled(true);

	window.setFramerateLimit(60);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;
	sf::Texture poleTexture;

	backgroundTexture.loadFromFile("resources/background.png");
	cartTexture.loadFromFile("resources/cart.png");
	poleTexture.loadFromFile("resources/pole.png");

	sf::Texture inputCartTexture;
	sf::Texture inputPoleTexture;

	inputCartTexture.loadFromFile("resources/inputCart.png");
	inputPoleTexture.loadFromFile("resources/inputPole.png");

	// --------------------------------------------------------------------

	sf::Sprite backgroundSprite;
	sf::Sprite cartSprite;
	sf::Sprite poleSprite;

	backgroundSprite.setTexture(backgroundTexture);
	cartSprite.setTexture(cartTexture);
	poleSprite.setTexture(poleTexture);

	backgroundSprite.setPosition(sf::Vector2f(0.0f, 0.0f));

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y)));
	poleSprite.setOrigin(sf::Vector2f(static_cast<float>(poleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(poleSprite.getTexture()->getSize().y)));

	sf::Sprite inputCartSprite;
	sf::Sprite inputPoleSprite;

	inputCartSprite.setTexture(inputCartTexture);
	inputPoleSprite.setTexture(inputPoleTexture);

	inputCartSprite.setOrigin(sf::Vector2f(static_cast<float>(inputCartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(inputCartSprite.getTexture()->getSize().y)));
	inputPoleSprite.setOrigin(sf::Vector2f(static_cast<float>(inputPoleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(inputPoleSprite.getTexture()->getSize().y)));

	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;
	float inputPixelsPerMeter = 8.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 40.0f;
	float cartMass = 2.0f;
	sf::Vector2f massPos(0.0f, poleLength);
	sf::Vector2f massVel(0.0f, 0.0f);
	float poleAngle = static_cast<float>(3.14159f) * 0.0f;
	float poleAngleVel = 0.0f;
	float poleAngleAccel = 0.0f;
	float cartX = 0.0f;
	float cartVelX = 0.0f;
	float cartAccelX = 0.0f;
	float poleRotationalFriction = 0.008f;
	float cartMoveRadius = 1.8f;
	float cartFriction = 0.02f;
	float maxSpeed = 3.0f;

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	float lowPassFitness = 0.0f;

	bool reverseDirection = false;

	bool trainMode = true;

	bool tDownLastFrame = false;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	sf::Font font;

	font.loadFromFile("resources/pixelated.ttf");

	sf::RenderTexture inputRT;

	inputRT.create(64, 32);

	float avgReward = 0.0f;
	float avgRewardDecay = 0.003f;

	float minReward = 0.0f;
	float maxReward = 1.0f;

	float totalTime = 0.0f;

	float plotUpdateTimer = 0.0f;

	htferl::HTFERL agent;

	std::vector<htfe::LayerDesc> layerDescs(4);

	layerDescs[0]._spatialWidth = 128;
	layerDescs[0]._spatialHeight = 128;
	layerDescs[0]._temporalWidth = 128;
	layerDescs[0]._temporalHeight = 128;

	layerDescs[1]._spatialWidth = 90;
	layerDescs[1]._spatialHeight = 90;
	layerDescs[1]._temporalWidth = 90;
	layerDescs[1]._temporalHeight = 90;

	layerDescs[2]._spatialWidth = 70;
	layerDescs[2]._spatialHeight = 70;
	layerDescs[2]._temporalWidth = 70;
	layerDescs[2]._temporalHeight = 70;

	layerDescs[3]._spatialWidth = 64;
	layerDescs[3]._spatialHeight = 64;
	layerDescs[3]._temporalWidth = 64;
	layerDescs[3]._temporalHeight = 64;

	std::vector<htferl::HTFERL::InputType> inputTypes(64 * 64, htferl::HTFERL::_state);

	std::uniform_int_distribution<int> actionXDist(0, 63);
	std::uniform_int_distribution<int> actionYDist(33, 63);

	std::vector<int> actionIndices;

	/*for (int x = 28; x < 34; x++)
	for (int y = 48; y < 54; y++) {
		if (inputTypes[x + y * 64] == htm::HTMRL::_action)
			continue;

		inputTypes[x + y * 64] = htm::HTMRL::_action;

		actionIndices.push_back(x + y * 64);
	}*/

	for (int i = 0; i < 1; i++) {
		int x = actionXDist(generator);
		int y = actionYDist(generator);

		if (inputTypes[x + y * 64] == htferl::HTFERL::_action)
			continue;

		inputTypes[x + y * 64] = htferl::HTFERL::_action;

		actionIndices.push_back(x + y * 64);
	}

	for (int i = 0; i < 8; i++) {
		int x = actionXDist(generator);
		int y = actionYDist(generator);

		if (inputTypes[x + y * 64] == htferl::HTFERL::_action)
			continue;

		inputTypes[x + y * 64] = htferl::HTFERL::_q;
	}

	agent.createRandom(cs, program, 64, 64, layerDescs, inputTypes, 7, -0.4f, 0.4f, generator);

	sf::RenderTexture htmRT;
	htmRT.create(1024, 1024, false);

	vis::HTFERLVisualizer visualizer;
	visualizer.create(1024);

	vis::Plot plot;

	plot._curves.resize(1);
	
	sf::RenderTexture plotRT;
	plotRT.create(800, 600, false);

	sf::Texture lineGradient;
	lineGradient.loadFromFile("resources/lineGradient.png");

	sf::Font tickFont;
	tickFont.loadFromFile("resources/arial.ttf");

	const int plotSampleTicks = 60;
	int plotSampleCounter = 0;
	
	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// Update fitness
		if (poleAngle < static_cast<float>(3.14159f))
			fitness = -(static_cast<float>(3.14159f)* 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(3.14159f)* 0.5f - (static_cast<float>(3.14159f)* 2.0f - poleAngle));

		//fitness += static_cast<float>(3.14159f)* 0.5f;

		//fitness = fitness - std::abs(poleAngleVel * 1.0f);

		//fitness = -std::abs(cartX);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			fitness = -cartX;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			fitness = cartX;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		//reward = dFitness * 5.0f;

		reward = fitness;

		if (totalTime == 0.0f)
			avgReward = reward;
		else
			avgReward = (1.0f - avgRewardDecay) * avgReward + avgRewardDecay * reward;

		minReward = std::min<float>(minReward, avgReward);
		maxReward = std::max<float>(maxReward, avgReward);

		if (plotSampleCounter == plotSampleTicks) {
			plotSampleCounter = 0;

			vis::Point p;
			p._position.x = plot._curves[0]._points.size() - 1;
			p._position.y = avgReward;
			p._color = sf::Color::Red;

			plot._curves[0]._points.push_back(p);
		}

		plotSampleCounter++;

		sf::Image img = inputRT.getTexture().copyToImage();

		for (int x = 0; x < 64; x++)
		for (int y = 0; y < 32; y++) {
			agent.setInput(x, y, img.getPixel(x, y).r / 255.0f);
		}

		agent.step(cs, reward * 0.005f, 0.5f, 0.995f, 0.05f, 0.05f, 0.1f, 0.1f, 0.005f, 0.005f, 0.1f, 1.0f, 400, 200, generator);

		float output = 0.0f;
		int c = 0;

		for (int i = 0; i < actionIndices.size(); i++) {
			output += agent.getOutput(i);
			c++;
		}

		output /= c;

		float dir = std::min<float>(1.0f, std::max<float>(-1.0f, 1.4f * (output)));

		//std::cout << dir << std::endl;

		float agentForce = 4000.0f * dir;
	
		prevFitness = fitness;

		// ---------------------------- Physics ----------------------------

		float pendulumCartAccelX = cartAccelX;

		if (cartX < -cartMoveRadius)
			pendulumCartAccelX = 0.0f;
		else if (cartX > cartMoveRadius)
			pendulumCartAccelX = 0.0f;

		poleAngleAccel = pendulumCartAccelX * std::cos(poleAngle) + g * std::sin(poleAngle);
		poleAngleVel += -poleRotationalFriction * poleAngleVel + poleAngleAccel * dt;
		poleAngle += poleAngleVel * dt;

		massPos = sf::Vector2f(cartX + std::cos(poleAngle + static_cast<float>(3.14159f)* 0.5f) * poleLength, std::sin(poleAngle + static_cast<float>(3.14159f)* 0.5f) * poleLength);

		float force = 0.0f;

		if (std::abs(cartVelX) < maxSpeed) {
			force = std::max<float>(-4000.0f, std::min<float>(4000.0f, agentForce));

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
				force = -4000.0f;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
				force = 4000.0f;
		}

		if (cartX < -cartMoveRadius) {
			cartX = -cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}
		else if (cartX > cartMoveRadius) {
			cartX = cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}

		cartAccelX = 0.25f * (force + massMass * poleLength * poleAngleAccel * std::cos(poleAngle) - massMass * poleLength * poleAngleVel * poleAngleVel * std::sin(poleAngle)) / (massMass + cartMass);
		cartVelX += -cartFriction * cartVelX + cartAccelX * dt;
		cartX += cartVelX * dt;

		poleAngle = std::fmod(poleAngle, (2.0f * static_cast<float>(3.14159f)));

		if (poleAngle < 0.0f)
			poleAngle += static_cast<float>(3.14159f)* 2.0f;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			if (!tDownLastFrame) {
				trainMode = !trainMode;
			}

			tDownLastFrame = true;
		}
		else
			tDownLastFrame = false;

		// ---------------------------- Rendering ----------------------------

		// Render to input buffer
		inputRT.clear();

		inputCartSprite.setPosition(sf::Vector2f(inputRT.getSize().x * 0.5f + inputPixelsPerMeter * cartX, inputRT.getSize().y * 0.5f + 4.0f));

		inputRT.draw(inputCartSprite);

		inputPoleSprite.setPosition(inputCartSprite.getPosition() + sf::Vector2f(0.0f, -4.0f));
		inputPoleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(3.14159f) + 180.0f);

		inputRT.draw(inputPoleSprite);

		inputRT.display();

		window.clear();

		window.draw(backgroundSprite);

		cartSprite.setPosition(sf::Vector2f(800.0f * 0.5f + pixelsPerMeter * cartX, 600.0f * 0.5f + 3.0f));

		window.draw(cartSprite);

		poleSprite.setPosition(cartSprite.getPosition() + sf::Vector2f(0.0f, -45.0f));
		poleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(3.14159f) + 180.0f);

		window.draw(poleSprite);

		sf::Sprite inputSprite;

		inputSprite.setTexture(inputRT.getTexture());

		inputSprite.setPosition(0, 0);
		inputSprite.setScale(4.0f, 4.0f);

		window.draw(inputSprite);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::B)) {
			plotRT.setActive();
			plotRT.clear(sf::Color::White);

			plot.draw(plotRT, lineGradient, tickFont, 0.5f, sf::Vector2f(0.0f, plot._curves[0]._points.size()), sf::Vector2f(minReward, maxReward), sf::Vector2f(64.0f, 64.0f), sf::Vector2f(plot._curves[0]._points.size() / 10.0f, (maxReward - minReward) / 10.0f), 2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);

			plotRT.display();

			sf::Sprite plotSprite;
			plotSprite.setTexture(plotRT.getTexture());

			window.draw(plotSprite);
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::V)) {
			htmRT.setActive();
			htmRT.clear(sf::Color::White);

			visualizer.update(htmRT, sf::Vector2f(512.0f, 512.0f), sf::Vector2f(1.95f, 1.95f), cs, agent, generator);

			htmRT.display();

			sf::Sprite htmSprite;
			htmSprite.setTexture(htmRT.getTexture());

			htmSprite.setScale(0.8f, 0.8f);
			htmSprite.setOrigin(512, 512);
			htmSprite.setPosition(400.0f, 300.0f);

			window.draw(htmSprite);
		}

		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();

		totalTime += dt;
		plotUpdateTimer += dt;
	} while (!quit);

	return 0;
}