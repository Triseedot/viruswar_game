import asyncio
import logging
import sys
from os import getenv

import time

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters.callback_data import CallbackData

from typing import Optional

import game

TOKEN = getenv("BOT_TOKEN")

dp = Dispatcher()

color = ["🔵", "🔴"]

game_instance = {}
player_id = {}
player_name = {}
last_move_time = {}


class SelectionCallback(CallbackData, prefix="selection"):
    player: int


class MoveCallback(CallbackData, prefix="move"):
    x: int
    y: int


@dp.message(CommandStart())
async def command_start_handler(message: Message):
    await message.answer(f"/game для начала игры")


async def get_game_id(message: Message):
    return 1000000 * message.chat.id + message.message_id


async def get_selection_keyboard(game_id: Optional[int] = None):
    player_selection = InlineKeyboardBuilder()
    for i in range(2):
        if game_id is None:
            player_selection.button(text=f"Свободно", callback_data=SelectionCallback(player=i))
        elif player_id[game_id][i]:
            player_selection.button(text=player_name[game_id][i], callback_data=SelectionCallback(player=-1))
        else:
            player_selection.button(text=f"Свободно", callback_data=SelectionCallback(player=i))
    return player_selection.as_markup()


async def get_move_keyboard(game_id: int):
    field = InlineKeyboardBuilder()
    for x in range(game.height):
        for y in range(game.width):
            field.button(text=await game_instance[game_id].get(x, y), callback_data=MoveCallback(x=x, y=y))
    field.button(text="Сдаться", callback_data=MoveCallback(x=-1, y=-1))
    return field.as_markup()


async def get_game_text(game_id: int):
    return (
        f"<i>Партия {game_id}</i>\n{player_name[game_id][0]} против {player_name[game_id][1]}\n"
        f"<b>Ходит:</b> {player_name[game_id][game_instance[game_id].currentPlayer]}\n"
        f"<b>Осталось ходов:</b> {game_instance[game_id].movesLeft}"
    )


@dp.message(Command("game"))
async def command_game_handler(message: Message):
    answer_message = await message.answer("<i>Ожидаем игроков</i>", reply_markup=await get_selection_keyboard())
    game_id = await get_game_id(answer_message)
    player_id[game_id] = [None, None]
    player_name[game_id] = [None, None]
    last_move_time[game_id] = time.time()


@dp.callback_query(SelectionCallback.filter())
async def callbacks_selection(
        callback: types.CallbackQuery,
        callback_data: SelectionCallback
):
    game_id = await get_game_id(callback.message)
    if callback_data.player != -1:
        player_id[game_id][callback_data.player] = callback.from_user.id
        player_name[game_id][callback_data.player] = f"{color[callback_data.player]} {callback.from_user.first_name}"
        await callback.answer()
        await callback.message.edit_reply_markup(reply_markup=await get_selection_keyboard(game_id))
        if player_id[game_id][0] and player_id[game_id][1]:
            for i in [3, 2, 1]:
                await callback.message.edit_text(f"<b>Начало через:</b> {i}")
                await asyncio.sleep(1)
            game_instance[game_id] = game.Instance()
            await game_instance[game_id].setup()
            await callback.message.edit_text(await get_game_text(game_id), reply_markup=await get_move_keyboard(game_id))
    else:
        await callback.answer("Место занято")


@dp.callback_query(MoveCallback.filter())
async def callbacks_move(
        callback: types.CallbackQuery,
        callback_data: MoveCallback
):
    game_id = await get_game_id(callback.message)
    if callback.from_user.id != player_id[game_id][game_instance[game_id].currentPlayer]:
        await callback.answer("Сейчас не ваш ход")
        return
    now = time.time()
    if now - last_move_time[game_id] < 1:
        await callback.answer("Ходите не так быстро")
        return
    x = callback_data.x
    y = callback_data.y
    if x == -1:
        await callback.message.answer(
            text=f"<i>Партия {game_id}</i>\n{player_name[game_id][0]} против {player_name[game_id][1]}\n"
                 f"<b>Победил</b> {player_name[game_id][(game_instance[game_id].currentPlayer + 1) % 2]} "
                 f"<i>(Сдача партии)</i>"
        )
        await callback.message.delete()
        return
    if not await game_instance[game_id].move(x, y):
        await callback.answer("Ход не корректный")
        return
    if await game_instance[game_id].is_over():
        await callback.message.answer(
            text=f"<i>Партия {game_id}</i>\n{player_name[game_id][0]} против {player_name[game_id][1]}\n"
                 f"<b>Победил</b> {player_name[game_id][(game_instance[game_id].currentPlayer + 1) % 2]}"
        )
        await callback.message.delete()
        return
    await callback.message.edit_text(await get_game_text(game_id), reply_markup=await get_move_keyboard(game_id))
    last_move_time[game_id] = time.time()


async def main():
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
