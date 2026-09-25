import asyncio
import logging
import os
import sys
import time
from copy import deepcopy

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.types import Message
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv

import game
from ai_player import VirusBot

bot = VirusBot(
    "models/viruswar_v1.2.pkl",
    simulations=1024,
)

load_dotenv()
TOKEN = os.environ["BOT_TOKEN"]

dp = Dispatcher()

color = ["🟢", "🔴"]

game_instance: dict[int, game.Instance] = {}
player_id: dict[int, list[int | None]] = {}
player_name: dict[int, list[str | None]] = {}
last_move_time: dict[int, float] = {}
history: dict[int, list[game.Instance]] = {}
history_time: dict[int, int] = {}


class SelectionCallback(CallbackData, prefix="selection"):
    player: int


class MoveCallback(CallbackData, prefix="move"):
    x: int
    y: int


class HistoryCallback(CallbackData, prefix="history"):
    x: int
    y: int


@dp.message(CommandStart())
async def command_start_handler(message: Message):
    await message.answer("/game для начала игры\n/help для правил")


async def get_game_id(message: Message):
    print(message.text)
    print(message.message_id)
    return message.message_id


async def get_selection_keyboard(game_id: int | None = None):
    player_selection = InlineKeyboardBuilder()
    for i in range(2):
        if game_id is None:
            player_selection.button(
                text="Свободно", callback_data=SelectionCallback(player=i)
            )
        elif player_id[game_id][i]:
            current_name = player_name[game_id][i]
            assert current_name is not None
            player_selection.button(
                text=current_name, callback_data=SelectionCallback(player=-1)
            )
        else:
            player_selection.button(
                text="Свободно", callback_data=SelectionCallback(player=i)
            )
    player_selection.button(
        text="Отменить игру", callback_data=SelectionCallback(player=-2)
    )
    player_selection.button(
        text="Заполнить свободные ботом", callback_data=SelectionCallback(player=-3)
    )
    player_selection.adjust(2, 1, 1)
    return player_selection.as_markup()


async def get_move_keyboard(game_id: int):
    field = InlineKeyboardBuilder()
    for x in range(game.height):
        for y in range(game.width):
            field.button(
                text=game_instance[game_id].get(x, y),
                callback_data=MoveCallback(x=x, y=y),
            )
    field.button(text="Сдаться", callback_data=MoveCallback(x=-1, y=-1))
    return field.as_markup()


async def get_history_keyboard(game_id: int):
    field = InlineKeyboardBuilder()
    for x in range(game.height):
        for y in range(game.width):
            field.button(
                text=history[game_id][history_time[game_id]].get(x, y),
                callback_data=HistoryCallback(x=x, y=y),
            )
        print()
    field.button(text="Назад", callback_data=HistoryCallback(x=-1, y=-1))
    field.button(text="Вперед", callback_data=HistoryCallback(x=-2, y=-2))
    field.button(text="Назад на 10", callback_data=HistoryCallback(x=-3, y=-3))
    field.button(text="Вперед на 10", callback_data=HistoryCallback(x=-4, y=-4))
    field.adjust(*([8] * 10 + [2] * 2))
    return field.as_markup()


async def get_game_text(game_id: int):
    return (
        f"{player_name[game_id][0]} против {player_name[game_id][1]}\n"
        f"<b>Ходит:</b> {player_name[game_id][game_instance[game_id].currentPlayer]}\n"
        f"<b>Осталось ходов:</b> {game_instance[game_id].movesLeft}"
    )


async def ai_step(message: Message):
    game_id = await get_game_id(message)
    x, y = bot.get_move(game_instance[game_id])
    assert game_instance[game_id].move(x, y)
    history[game_id].append(deepcopy(game_instance[game_id]))
    if game_instance[game_id].is_over():
        await end_game(message, game_id)
        return
    if player_id[game_id][game_instance[game_id].currentPlayer] is None:
        await ai_step(message)
    else:
        await message.edit_text(
            await get_game_text(game_id), reply_markup=await get_move_keyboard(game_id)
        )


@dp.message(Command("help"))
async def command_help_handler(message: Message):
    await message.answer(
        'На поле 8 на 10 играют двое, представляя из себя враждующие "вирусы". Изначально у каждого по '
        'одной клетке, которые находятся в разных углах и имеют состояние "живых". Ходят по очереди, причем по 3 '
        'действия сразу, во время которых можно либо создать новую "живую клетку", но чтобы она была соседней с '
        'какой-то клеткой, что уже как-то последовательно соединена с уже существующей "живой", либо съесть "живую" '
        'клетку соперника соседней своей к ней, но та тоже должна быть последовательно соединена с "живой", '
        'делая при этом обретенную клетку собственной "мертвой". "Мертвые" клетки уже не подлежат пересъедению, '
        'они лишь служат "тропой", притом новых "живых" самостоятельно не могут образовывать. Соседними называются '
        "клетки, которые имеют общую сторону. Проигрывает тот, кто не может сделать ход."
    )


@dp.message(Command("game"))
async def command_game_handler(message: Message):
    answer_message = await message.answer(
        "<i>Ожидаем игроков</i>", reply_markup=await get_selection_keyboard()
    )
    game_id = await get_game_id(answer_message)
    game_instance[game_id] = game.Instance()
    player_id[game_id] = [None, None]
    player_name[game_id] = [None, None]
    last_move_time[game_id] = time.time()


@dp.callback_query(SelectionCallback.filter())
async def callbacks_selection(
    callback: types.CallbackQuery, callback_data: SelectionCallback
):
    if not isinstance(callback.message, Message):
        return
    game_id = await get_game_id(callback.message)
    if callback_data.player == -2:
        await callback.message.delete()
        del game_instance[game_id]
        del player_id[game_id]
        del player_name[game_id]
        del last_move_time[game_id]
    if callback_data.player >= 0:
        player_id[game_id][callback_data.player] = callback.from_user.id
        player_name[game_id][callback_data.player] = (
            f"{color[callback_data.player]} {callback.from_user.first_name}"
        )
        await callback.answer()
        await callback.message.edit_reply_markup(
            reply_markup=await get_selection_keyboard(game_id)
        )
        if player_id[game_id][0] and player_id[game_id][1]:
            history[game_id] = [deepcopy(game_instance[game_id])]
            await callback.message.edit_text(
                await get_game_text(game_id),
                reply_markup=await get_move_keyboard(game_id),
            )
    elif callback_data.player == -3:
        for i in range(2):
            if player_name[game_id][i] is None:
                player_name[game_id][i] = f"{color[i]} Бот"
        for i in [3, 2, 1]:
            await callback.message.edit_text(f"<b>Начало через:</b> {i}")
            await asyncio.sleep(1)
        history[game_id] = [deepcopy(game_instance[game_id])]
        await callback.message.edit_text(
            await get_game_text(game_id),
            reply_markup=await get_move_keyboard(game_id),
        )
        if player_id[game_id][game_instance[game_id].currentPlayer] is None:
            await ai_step(callback.message)
    else:
        await callback.answer("Место занято")


async def end_game(message, game_id: int, text: str = ""):
    await message.edit_text(
        f"{player_name[game_id][0]} против {player_name[game_id][1]}\n"
        f"<b>Победил</b> {player_name[game_id][(game_instance[game_id].currentPlayer + 1) % 2]} "
        + text
    )
    history_time[game_id] = len(history[game_id]) - 1
    await message.edit_reply_markup(reply_markup=await get_history_keyboard(game_id))
    del game_instance[game_id]
    del player_id[game_id]
    del player_name[game_id]
    del last_move_time[game_id]


@dp.callback_query(MoveCallback.filter())
async def callbacks_move(callback: types.CallbackQuery, callback_data: MoveCallback):
    if not isinstance(callback.message, Message):
        return
    game_id = await get_game_id(callback.message)
    if (
        callback.from_user.id
        != player_id[game_id][game_instance[game_id].currentPlayer]
    ):
        await callback.answer("Сейчас не ваш ход")
        return
    now = time.time()
    if now - last_move_time[game_id] < 0.2:
        await callback.answer("Ходите не так быстро")
        return
    x = callback_data.x
    y = callback_data.y
    if x == -1:
        await end_game(callback.message, game_id, "(Сдача партии)")

        return
    if not game_instance[game_id].move(x, y):
        await callback.answer("Ход не корректный")
        return
    if game_instance[game_id].is_over():
        await end_game(callback.message, game_id)
        return
    last_move_time[game_id] = time.time()
    history[game_id].append(deepcopy(game_instance[game_id]))
    if player_id[game_id][game_instance[game_id].currentPlayer] is None:
        await ai_step(callback.message)
    else:
        await callback.message.edit_text(
            await get_game_text(game_id), reply_markup=await get_move_keyboard(game_id)
        )


@dp.callback_query(HistoryCallback.filter())
async def callbacks_history(
    callback: types.CallbackQuery, callback_data: HistoryCallback
):
    if not isinstance(callback.message, Message):
        return
    game_id = await get_game_id(callback.message)
    x = callback_data.x
    if x == -1:
        history_time[game_id] = max(history_time[game_id] - 1, 0)
    elif x == -2:
        history_time[game_id] = min(history_time[game_id] + 1, len(history[game_id]))
    elif x == -3:
        history_time[game_id] = max(history_time[game_id] - 10, 0)
    elif x == -4:
        history_time[game_id] = min(history_time[game_id] + 10, len(history[game_id]))

    await callback.message.edit_reply_markup(
        reply_markup=await get_history_keyboard(game_id)
    )
    await callback.answer()


async def main():
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
