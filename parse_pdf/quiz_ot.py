import logging
from telegram import Update, Poll
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    PollHandler,
)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Quiz Questions ---
QUIZ_QUESTIONS = [
    {
        "question": "What is the capital of France?",
        "options": ["Berlin", "Madrid", "Paris", "Rome"],
        "correct_option_id": 2,
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "options": ["Earth", "Mars", "Jupiter", "Venus"],
        "correct_option_id": 1,
    },
    {
        "question": "What is the largest mammal in the world?",
        "options": ["Elephant", "Blue Whale", "Giraffe", "Great White Shark"],
        "correct_option_id": 1,
    },
    # Add more questions if you like...
]

# --- Command Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to the Quiz Bot!\nSend /quiz to start a new quiz."
    )

async def quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("--- DEBUG: /quiz command received. Starting new quiz. ---") # DEBUG
    context.user_data["current_question"] = 0
    context.user_data["score"] = 0
    await send_question(update.effective_chat.id, context)

async def send_question(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    question_index = context.user_data.get("current_question", 0)
    print(f"--- DEBUG: In send_question for question index: {question_index} ---") # DEBUG
    
    if question_index < len(QUIZ_QUESTIONS):
        question_data = QUIZ_QUESTIONS[question_index]
        print(f"--- DEBUG: Sending poll for question: '{question_data['question']}' ---") # DEBUG
        message = await context.bot.send_poll(
            chat_id=chat_id,
            question=question_data["question"],
            options=question_data["options"],
            type=Poll.QUIZ,
            correct_option_id=question_data["correct_option_id"],
            is_anonymous=False,
            explanation=f"The correct answer is {question_data['options'][question_data['correct_option_id']]}."
        )
        # Save chat_id to find it again in the poll handler
        context.bot_data[message.poll.id] = {
            "chat_id": chat_id,
            "correct_option_id": question_data["correct_option_id"],
        }
        print(f"--- DEBUG: Poll sent. Stored data for poll ID: {message.poll.id} ---") # DEBUG
    else:
        # If quiz is over, show the result
        print("--- DEBUG: Quiz is over. Calling show_result. ---") # DEBUG
        await show_result(chat_id, context)

async def receive_poll_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("\n--- DEBUG: Poll update received! Entering receive_poll_update. ---") # DEBUG
    poll = update.poll
    
    if not poll.is_closed:
        print("--- DEBUG: Poll is not closed yet. Exiting. ---") # DEBUG
        return

    print(f"--- DEBUG: Poll {poll.id} is closed. Proceeding... ---") # DEBUG
    
    try:
        poll_data = context.bot_data[poll.id]
        chat_id = poll_data["chat_id"]
        correct_option = poll_data["correct_option_id"]
        print(f"--- DEBUG: Found data for this poll. Chat ID is {chat_id}. ---") # DEBUG
    except KeyError:
        print(f"--- DEBUG: ERROR! Could not find data for poll ID {poll.id}. Cannot continue quiz. ---") # DEBUG
        return

    # Update score
    if poll.options[correct_option]["voter_count"] == 1:
        context.user_data["score"] = context.user_data.get("score", 0) + 1
        print(f"--- DEBUG: User answered correctly. New score: {context.user_data['score']} ---") # DEBUG
    else:
        print("--- DEBUG: User answered incorrectly. ---") # DEBUG

    # Move to the next question
    context.user_data["current_question"] += 1
    print(f"--- DEBUG: Incremented question index to: {context.user_data['current_question']} ---") # DEBUG
    
    # Send the next question or show the final result
    await send_question(chat_id, context)


async def show_result(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    score = context.user_data.get("score", 0)
    total_questions = len(QUIZ_QUESTIONS)
    print(f"--- DEBUG: Showing final score: {score}/{total_questions} ---") # DEBUG
    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            f"Quiz finished! Your final score is {score}/{total_questions}.\n\n"
            "Send /quiz to play again!"
        )
    )
    context.user_data.clear()

def main() -> None:
    """Run the bot."""
    application = Application.builder().token("7567341341:AAEXTCDvocJVxm-hdQR2p-98kk9bdZ3q504").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("quiz", quiz))
    application.add_handler(PollHandler(receive_poll_update))

    print("Bot is running... Press Ctrl-C to stop.")
    application.run_polling()

if __name__ == "__main__":
    main()