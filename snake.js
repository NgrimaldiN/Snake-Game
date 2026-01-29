const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

const GRID_SIZE = 60; // Size of each square in pixels
const TILE_COUNT_X = 10;
const TILE_COUNT_Y = 9;

const COLOR_LIGHT = '#AAD751';
const COLOR_DARK = '#A2D149';
const SNAKE_COLOR = '#437DD4';
const HAS_EYES = true;
const APPLE_COLOR = '#E91E63';
const WALL_COLOR = '#578a34'; // Darker green for walls
const WALL_SPAWN_CHANCE = 0.6; // 60% chance to spawn wall

// Load assets
const appleImg = new Image();
appleImg.src = 'images/apple.png';

const headUpImg = new Image(); headUpImg.src = 'images/head_up.png';
const headDownImg = new Image(); headDownImg.src = 'images/head_down.png';
const headLeftImg = new Image(); headLeftImg.src = 'images/head_left.png';
const headRightImg = new Image(); headRightImg.src = 'images/head_right.png';

const tailUpImg = new Image(); tailUpImg.src = 'images/tail_up.png';
const tailDownImg = new Image(); tailDownImg.src = 'images/tail_down.png';
const tailLeftImg = new Image(); tailLeftImg.src = 'images/tail_left.png';
const tailRightImg = new Image(); tailRightImg.src = 'images/tail_right.png';

const bodyVerticalImg = new Image(); bodyVerticalImg.src = 'images/body_vertical.png';
const bodyHorizontalImg = new Image(); bodyHorizontalImg.src = 'images/body_horizontal.png';

const bodyTopLeftImg = new Image(); bodyTopLeftImg.src = 'images/body_topleft.png';
const bodyTopRightImg = new Image(); bodyTopRightImg.src = 'images/body_topright.png';
const bodyBottomLeftImg = new Image(); bodyBottomLeftImg.src = 'images/body_bottomleft.png';
const bodyBottomRightImg = new Image(); bodyBottomRightImg.src = 'images/body_bottomright.png';

const trophyImg = new Image();
trophyImg.src = 'images/trophy.png';

// Game state
let score = 0;
let highScore = localStorage.getItem('snakeHighScore') || 0;
let gameLoopId;
let isGameOver = false;

// Snake state
// Start in the middle
let snake = [];
let dx = 1;
let dy = 0;
let inputQueue = [];

// Apple and Wall state
let apples = []; // multiple apples
let walls = [];
let applesEatenTotal = 0; // Track for wall spawning pattern

// Game loop timing
let lastTime = 0;
let currentSpeed = 135; // Default Normal (135ms)

// Update High Score Display
const highScoreEl = document.getElementById('high-score');
if (highScoreEl) highScoreEl.innerText = `🏆 ${highScore}`;

function spawnWall() {
	// Deterministic spawning based on apple count (handled in update)
	// Deterministic spawning based on apple count (handled in update)
	// Removed random chance to strictly follow 'every other apple' rule


	let valid = false;
	let wallCandidate = { x: 0, y: 0 };
	let attempts = 0;

	while (!valid && attempts < 50) {
		attempts++;
		wallCandidate.x = Math.floor(Math.random() * TILE_COUNT_X);
		wallCandidate.y = Math.floor(Math.random() * TILE_COUNT_Y);

		valid = true;

		// 1. Check collision with snake
		for (let segment of snake) {
			if (segment.x === wallCandidate.x && segment.y === wallCandidate.y) {
				valid = false; break;
			}
		}

		// 2. Check collision with existing walls & 1-tile gap rule
		if (valid) {
			for (let w of walls) {
				if (Math.abs(w.x - wallCandidate.x) <= 1 && Math.abs(w.y - wallCandidate.y) <= 1) {
					valid = false; break;
				}
			}
		}

		// 3. Prevent blocking the head immediately (safety)
		// Wiki Rule 2: 3 blocks taxicab distance buffer
		const head = snake[0];
		if (valid && head) {
			const dist = Math.abs(wallCandidate.x - head.x) + Math.abs(wallCandidate.y - head.y);
			if (dist <= 3) valid = false;
		}

		// 4. Don't spawn on apples
		if (valid) {
			for (let a of apples) {
				if (a.x === wallCandidate.x && a.y === wallCandidate.y) {
					valid = false; break;
				}
			}
		}

		// 5. Corner Protection (Wiki rule)
		if (valid) {
			const corners = [
				{ x: 0, y: 0 }, { x: TILE_COUNT_X - 1, y: 0 },
				{ x: 0, y: TILE_COUNT_Y - 1 }, { x: TILE_COUNT_X - 1, y: TILE_COUNT_Y - 1 }
			];
			for (let c of corners) {
				if (Math.abs(wallCandidate.x - c.x) <= 1 && Math.abs(wallCandidate.y - c.y) <= 1) {
					valid = false; break;
				}
			}
		}

		// 6. Dead End / Trap Prevention (Degree Check)
		// Ensure that placing this wall doesn't reduce any neighbor's exits to < 2
		if (valid) {
			const dirs = [{ x: 0, y: 1 }, { x: 0, y: -1 }, { x: 1, y: 0 }, { x: -1, y: 0 }];

			// Check all 4 neighbors of the candidate wall
			for (let d of dirs) {
				const nx = wallCandidate.x + d.x;
				const ny = wallCandidate.y + d.y;

				// If neighbor is valid board tile and NOT a wall/snake
				if (nx >= 0 && nx < TILE_COUNT_X && ny >= 0 && ny < TILE_COUNT_Y) {
					// Is this neighbor occupied?
					let occupied = false;
					for (let w of walls) if (w.x === nx && w.y === ny) occupied = true;
					for (let s of snake) if (s.x === nx && s.y === ny) occupied = true;

					if (!occupied) {
						// Calculate Degree of this empty neighbor
						// It needs at least 2 exits to not be a dead end/trap
						let exits = 0;
						for (let d2 of dirs) {
							const nnx = nx + d2.x;
							const nny = ny + d2.y;

							// Check if NN is valid exit
							if (nnx >= 0 && nnx < TILE_COUNT_X && nny >= 0 && nny < TILE_COUNT_Y) {
								let nnOccupied = false;
								// It is blocked if it is EXISTING wall/snake OR the NEW candidate
								if (nnx === wallCandidate.x && nny === wallCandidate.y) nnOccupied = true;
								for (let w of walls) if (w.x === nnx && w.y === nny) nnOccupied = true;
								for (let s of snake) if (s.x === nnx && s.y === nny) nnOccupied = true;

								if (!nnOccupied) exits++;
							}
						}

						if (exits < 2) {
							valid = false; break; // Found a neighbor that would become a trap
						}
					}
				}
			}
		}

		// 7. Connectivity Check (Flood Fill)
		if (valid) {
			if (typeof isMapConnected === 'function' && !isMapConnected(snake[0], apples, walls, wallCandidate)) {
				valid = false;
			}
		}
	}

	if (valid) {
		walls.push({ x: wallCandidate.x, y: wallCandidate.y });
	}
}

function spawnApple() {
	// Ensure we have 5 apples
	let attempts = 0;
	while (apples.length < 5 && attempts < 100) {
		attempts++;
		let candidate = {
			x: Math.floor(Math.random() * TILE_COUNT_X),
			y: Math.floor(Math.random() * TILE_COUNT_Y)
		};

		let valid = true;

		// Check if spawned on snake
		for (let segment of snake) {
			if (segment.x === candidate.x && segment.y === candidate.y) {
				valid = false; break;
			}
		}

		// Check if spawned on wall
		if (valid) {
			for (let w of walls) {
				if (w.x === candidate.x && w.y === candidate.y) {
					valid = false; break;
				}
			}
		}

		// Check if spawned on other apples
		if (valid) {
			for (let a of apples) {
				if (a.x === candidate.x && a.y === candidate.y) {
					valid = false; break;
				}
			}
		}

		if (valid) {
			apples.push(candidate);
		}
	}
}

function processInput() {
	if (inputQueue.length > 0) {
		const input = inputQueue.shift();
		const potentialDx = input.dx;
		const potentialDy = input.dy;

		// Prevent reversing direction
		if (potentialDx !== -dx && potentialDy !== -dy) {
			dx = potentialDx;
			dy = potentialDy;
		}
	}
}

function update() {
	if (isGameOver) return;

	processInput();

	const head = { x: snake[0].x + dx, y: snake[0].y + dy };

	// Wall Collision Check (Boundary)
	if (head.x < 0 || head.x >= TILE_COUNT_X || head.y < 0 || head.y >= TILE_COUNT_Y) {
		gameOver();
		return;
	}

	// Wall Collision Check (Obstacles)
	for (let w of walls) {
		if (head.x === w.x && head.y === w.y) {
			gameOver();
			return;
		}
	}

	// Self Collision Check
	for (let i = 0; i < snake.length; i++) {
		if (head.x === snake[i].x && head.y === snake[i].y) {
			gameOver();
			return;
		}
	}

	snake.unshift(head); // Add new head

	// Apple Eating
	let eatenIndex = -1;
	for (let i = 0; i < apples.length; i++) {
		if (head.x === apples[i].x && head.y === apples[i].y) {
			eatenIndex = i;
			break;
		}
	}

	if (eatenIndex !== -1) {
		// Ate an apple
		apples.splice(eatenIndex, 1);
		score++;
		applesEatenTotal++;

		const scoreEl = document.getElementById('score');
		if (scoreEl) scoreEl.innerText = score.toString();

		if (score > highScore) {
			highScore = score;
			localStorage.setItem('snakeHighScore', highScore);
			if (highScoreEl) highScoreEl.innerText = `🏆 ${highScore}`;
		}

		// Spawn Wall Logic
		// 1st apple -> Spawn
		// Then every 2 apples (3, 5, 7...)
		if (applesEatenTotal === 1 || (applesEatenTotal > 1 && (applesEatenTotal - 1) % 2 === 0)) {
			spawnWall();
		}

		spawnApple();
	} else {
		snake.pop(); // Remove tail
	}
}

function draw() {
	// 1. Draw Checkerboard Background
	for (let y = 0; y < TILE_COUNT_Y; y++) {
		for (let x = 0; x < TILE_COUNT_X; x++) {
			ctx.fillStyle = (x + y) % 2 === 0 ? COLOR_LIGHT : COLOR_DARK;
			ctx.fillRect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE);
		}
	}

	// 2. Draw Walls
	ctx.fillStyle = WALL_COLOR;
	for (let w of walls) {
		ctx.fillRect(w.x * GRID_SIZE, w.y * GRID_SIZE, GRID_SIZE, GRID_SIZE);
		// Add a little detail to look like a bush/wall
		ctx.fillStyle = 'rgba(0,0,0,0.2)';
		ctx.fillRect((w.x * GRID_SIZE) + 5, (w.y * GRID_SIZE) + 5, GRID_SIZE - 10, GRID_SIZE - 10);
		ctx.fillStyle = WALL_COLOR;
	}

	// 3. Draw Apples
	for (let a of apples) {
		if (appleImg.complete && appleImg.naturalWidth !== 0) {
			ctx.drawImage(appleImg, a.x * GRID_SIZE, a.y * GRID_SIZE, GRID_SIZE, GRID_SIZE);
		} else {
			ctx.fillStyle = APPLE_COLOR;
			ctx.beginPath();
			ctx.arc((a.x + 0.5) * GRID_SIZE, (a.y + 0.5) * GRID_SIZE, GRID_SIZE * 0.4, 0, Math.PI * 2);
			ctx.fill();
		}
	}

	// 3. Draw Snake
	ctx.fillStyle = SNAKE_COLOR;
	const eyeSize = GRID_SIZE * 0.15;
	const eyeOffset = GRID_SIZE * 0.2;

	for (let i = 0; i < snake.length; i++) {
		const seg = snake[i];
		const segX = seg.x * GRID_SIZE;
		const segY = seg.y * GRID_SIZE;

		if (i === 0) {
			// Head
			let img = headRightImg;
			if (dx === 1) img = headRightImg;
			else if (dx === -1) img = headLeftImg;
			else if (dy === 1) img = headDownImg;
			else if (dy === -1) img = headUpImg;

			ctx.drawImage(img, segX, segY, GRID_SIZE, GRID_SIZE);

		} else if (i === snake.length - 1) {
			// Tail
			const prev = snake[i - 1]; // Segment towards the head
			let img = tailRightImg;

			// Determine direction based on previous segment
			if (prev.x > seg.x) img = tailLeftImg; // Tail points left (connected to right)
			else if (prev.x < seg.x) img = tailRightImg;
			else if (prev.y > seg.y) img = tailUpImg;
			else if (prev.y < seg.y) img = tailDownImg;

			ctx.drawImage(img, segX, segY, GRID_SIZE, GRID_SIZE);

		} else {
			// Body
			const prev = snake[i - 1]; // Towards head
			const next = snake[i + 1]; // Towards tail
			let img = bodyHorizontalImg;

			// Simple check for straight vs corner
			if (prev.x === next.x) {
				img = bodyVerticalImg;
			} else if (prev.y === next.y) {
				img = bodyHorizontalImg;
			} else {
				// Corner
				// We need to check which neighbors we have
				// Example: If we have a neighbor to the Right (x+1) and Down (y+1), we use TopLeft?
				// Naming convention usually implies the "filled" part or the "outer corner".
				// Let's assume:
				// body_topleft connects Left and Top? No, that would be a curve going from Left to Top.
				// Let's deduce from typical sprite sheet logic or assume standard curves.
				// body_bottomleft: Connects Bottom and Left.
				// body_bottomright: Connects Bottom and Right.
				// body_topleft: Connects Top and Left.
				// body_topright: Connects Top and Right.

				const hasLeft = (prev.x < seg.x || next.x < seg.x);
				const hasRight = (prev.x > seg.x || next.x > seg.x);
				const hasUp = (prev.y < seg.y || next.y < seg.y);
				const hasDown = (prev.y > seg.y || next.y > seg.y);

				if (hasDown && hasLeft) img = bodyBottomLeftImg;
				else if (hasDown && hasRight) img = bodyBottomRightImg;
				else if (hasUp && hasLeft) img = bodyTopLeftImg;
				else if (hasUp && hasRight) img = bodyTopRightImg;
			}

			ctx.drawImage(img, segX, segY, GRID_SIZE, GRID_SIZE);
		}
	}
}

function gameLoop(currentTime) {
	if (!lastTime) lastTime = currentTime;
	const deltaTime = currentTime - lastTime;

	if (deltaTime >= currentSpeed) {
		update();
		lastTime = currentTime;
	}

	draw();

	if (!isGameOver) {
		gameLoopId = requestAnimationFrame(gameLoop);
	}
}

function handleKey(e) {
	// Prevent default scrolling for arrow keys
	if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " "].indexOf(e.key) > -1) {
		e.preventDefault();
	}

	if (isGameOver && (e.key === 'Enter' || e.key === ' ')) {
		resetGame();
		return;
	}

	// prevent adding too many moves to queue
	if (inputQueue.length >= 2) return;

	// Based on the last input in queue, or current direction if queue empty
	let lastDx = dx;
	let lastDy = dy;
	if (inputQueue.length > 0) {
		lastDx = inputQueue[inputQueue.length - 1].dx;
		lastDy = inputQueue[inputQueue.length - 1].dy;
	}

	switch (e.key) {
		case 'ArrowUp':
			if (lastDy !== 1) inputQueue.push({ dx: 0, dy: -1 });
			break;
		case 'ArrowDown':
			if (lastDy !== -1) inputQueue.push({ dx: 0, dy: 1 });
			break;
		case 'ArrowLeft':
			if (lastDx !== 1) inputQueue.push({ dx: -1, dy: 0 });
			break;
		case 'ArrowRight':
			if (lastDx !== -1) inputQueue.push({ dx: 1, dy: 0 });
			break;
	}
}

function gameOver() {
	isGameOver = true;
	cancelAnimationFrame(gameLoopId);
	document.getElementById('game-over-screen').style.display = 'flex';
	document.getElementById('final-score').innerText = `Score: ${score}`;
}

function resetGame() {
	score = 0;
	const scoreEl = document.getElementById('score');
	if (scoreEl) scoreEl.innerText = "0";

	snake = [
		{ x: 2, y: 4 },
		{ x: 1, y: 4 },
		{ x: 0, y: 4 }
	];
	dx = 1;
	dy = 0;
	inputQueue = [];
	isGameOver = false;
	walls = []; // Clear walls
	applesEatenTotal = 0;

	// Fixed initial apple positions (Wiki: Cross pattern for first 5)
	apples = [
		{ x: 4, y: 2 },
		{ x: 8, y: 2 },
		{ x: 6, y: 4 },
		{ x: 4, y: 6 },
		{ x: 8, y: 6 }
	];
	// spawnApple(); // Don't random spawn on reset, use fixed
	document.getElementById('game-over-screen').style.display = 'none';
	lastTime = 0;
	gameLoopId = requestAnimationFrame(gameLoop);
}

// Flood Fill Helper to ensure fairness
function isMapConnected(startNode, targets, currentWalls, newWall) {
	const obstacles = new Set();
	// Add walls
	for (let w of currentWalls) obstacles.add(`${w.x},${w.y}`);
	// Add new wall
	if (newWall) obstacles.add(`${newWall.x},${newWall.y}`);
	// Add snake body
	if (typeof snake !== 'undefined') {
		for (let s of snake) obstacles.add(`${s.x},${s.y}`);
	}

	const visited = new Set();
	const queue = [];

	if (!startNode) return true;

	queue.push(startNode);
	visited.add(`${startNode.x},${startNode.y}`);

	// Identify distinct target keys
	const targetKeys = new Set();
	for (let t of targets) targetKeys.add(`${t.x},${t.y}`);

	let applesFound = 0;

	while (queue.length > 0) {
		const curr = queue.shift();
		const k = `${curr.x},${curr.y}`;

		if (targetKeys.has(k)) {
			applesFound++;
			targetKeys.delete(k);
		}

		if (targetKeys.size === 0) return true; // Found all

		const dirs = [{ x: 0, y: 1 }, { x: 0, y: -1 }, { x: 1, y: 0 }, { x: -1, y: 0 }];
		for (let d of dirs) {
			const nx = curr.x + d.x;
			const ny = curr.y + d.y;
			const nk = `${nx},${ny}`;

			// Bound check
			if (nx >= 0 && nx < TILE_COUNT_X && ny >= 0 && ny < TILE_COUNT_Y) {
				if (!obstacles.has(nk) && !visited.has(nk)) {
					visited.add(nk);
					queue.push({ x: nx, y: ny });
				}
			}
		}
	}

	return targetKeys.size === 0;
}

// Init
window.addEventListener('keydown', handleKey);

const restartBtn = document.getElementById('restart-btn');
if (restartBtn) restartBtn.addEventListener('click', resetGame);

// Start
// UI Handlers
const settingsBtn = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const closeSettingsBtn = document.getElementById('close-settings');
const speedOptions = document.querySelectorAll('.speed-option');

if (settingsBtn) {
	settingsBtn.addEventListener('click', () => {
		settingsModal.style.display = 'flex';
	});
}

if (closeSettingsBtn) {
	closeSettingsBtn.addEventListener('click', () => {
		settingsModal.style.display = 'none';
		// focus back on window for keys
		window.focus();
	});
}

speedOptions.forEach(opt => {
	opt.addEventListener('click', () => {
		// Update UI
		speedOptions.forEach(o => o.classList.remove('selected'));
		opt.classList.add('selected');

		// Set Speed
		const speed = opt.getAttribute('data-speed');
		if (speed === 'normal') currentSpeed = 135;
		else if (speed === 'fast') currentSpeed = 89.1;

		// Reset game to apply new flow feel immediately ? 
		// Or just let it speed up. Let's just update speed.
		// User typically expects reset on major setting change but Google Snake often just changes.
		// Let's reset to ensure fair start.
		resetGame();
		settingsModal.style.display = 'none';
	});
});

// Start
resetGame();
