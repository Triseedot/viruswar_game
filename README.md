# Virus War

Virus War is a turn-based strategy game played in Telegram. Challenge another player or play against a computer opponent on an 8 × 10 board.

**Bot:** [@viruswar_game_bot](https://t.me/viruswar_game_bot). The bot's interface is in Russian.

## Rules

Each player starts with one live cell in an opposite corner. Green moves first. Players take turns making three actions each: place a live cell on an empty square, or capture an opponent's live cell and turn it into one of your dead cells.

You can act on a square adjacent to your connected cells. Connections extend through your live and dead cells, but must lead back to a live cell. Only cells sharing a side are adjacent; diagonal connections do not count. Dead cells cannot be captured again. If you have no legal move, you lose.

🟢 and 🔴 represent live cells; 🟩 and 🟥 represent dead cells.

## Computer opponent

The computer opponent uses a neural network to evaluate positions and suggest moves, then searches possible continuations before choosing an action. The network is implemented with JAX and Flax; the search uses MCTX. Its model weights are stored in `models/`.

## Installation

To run your own instance, obtain a bot token from [BotFather](https://t.me/BotFather), then clone the repository and install its dependencies:

```bash
git clone https://github.com/Triseedot/viruswar_game.git
cd viruswar_game
pip install -r requirements.txt
```

Create a `.env` file in the project directory containing `BOT_TOKEN=your_token`, then start the bot with `python main.py`.

## Usage

Send `/game` in a chat to create a game. A second player can join by selecting an empty slot, or you can choose **Заполнить свободные ботом** to play against the computer. Make moves by tapping cells on the board. You can surrender during a game and browse recorded positions after it ends.

## Commands

- `/start` — show the available commands.
- `/help` — show the rules in Russian.
- `/game` — create a game.
