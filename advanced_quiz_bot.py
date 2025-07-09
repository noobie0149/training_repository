import logging
import os
import re
from dotenv import load_dotenv
from telegram import Update, Poll, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, PollAnswerHandler, ContextTypes, CallbackQueryHandler
from pinecone import Pinecone
import google.generativeai as genai
import json

# --- 1. SETUP AND INITIALIZATION ---

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Client Initialization ---
# Load credentials from .env file
BOT_TOKEN = os.getenv("attemptOneBot_token")
PINECONE_API_KEY = os.getenv("pinecone_api")
GEMINI_API_KEY = os.getenv("gemma_gemini_api")

# Initialize Pinecone and Gemini
pc = Pinecone(api_key=PINECONE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)


# --- Pinecone and Gemini Model Setup ---
model = genai.GenerativeModel('gemini-2.5-flash')
index_name = "biology"
logging.info(f"Connected to Pinecone for index: {index_name}")
logging.info("Gemini Model initialized.")

# --- LOGGING FOR QUIZ OUTCOMES ---
LOG_FILE = "boot.log"
def log_quiz_result(user_id, chat_id, username, quiz_data):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            "user_id": user_id,
            "chat_id": chat_id,
            "username": username,
            **quiz_data
        }, ensure_ascii=False) + "\n")

async def generate_search_queries(user_query: str) -> list[str]:
    """
    Generates conceptual search queries based on the user's topic.
    """
    system_prompt = f"""
    You are a sophisticated query generation expert for a vector database. Your task is to analyze the user's question and generate a conceptual search query.

    **Instructions:**

    1.  **Identify the Core Question:** First, understand the main subject of the question stem.
    2.  **Generate Conceptual Queries:** Depending on the type of question, create concise, conceptual query..
    3.  **Output Format:**
        * Do not number the queries or use bullet points.
        * Present each conceptual query on a new line.
    4.  ** In each query you generate do not assume any prior knowledge on the subject when generating the queries, simply frame the queries for the alterantives in conjunction with what the question is trying to ask.
    5.  ** The number of queries generated is dependant on the type of alternative questions.
    
    **User Question:** "{user_query}"

    **Conceptual Search Queries:**
    """
    try:
        response = await model.generate_content_async(system_prompt)
        queries = [query.strip() for query in response.text.strip().split('\n') if query.strip()]
        logging.info(f"Original query: '{user_query}' | Generated {len(queries)} conceptual queries: {queries}")
        if not queries:
             raise ValueError("Model failed to generate queries.")
        return queries
    except Exception as e:
        logging.error(f"Error during conceptual query generation: {e}")
        return [user_query]

async def process_query_for_context(user_query: str) -> tuple[str, list]:
    """
    Orchestrates the query workflow to retrieve context from Pinecone.
    Returns both the context and the source information.
    """
    search_queries = await generate_search_queries(user_query)
    dense_index = pc.Index(index_name)
    sources = []  # Store source information

    try:
        index_stats = dense_index.describe_index_stats()
        namespaces = list(index_stats.namespaces.keys())
    except Exception as e:
        logging.error(f"Could not connect to Pinecone index '{index_name}'. Error: {e}")
        return "", []

    all_hits = []
    seen_ids = set()

    try:
        for query in search_queries:
            logging.info(f"Searching with conceptual query: '{query}'")
            for ns in namespaces:
                results = dense_index.search(
                    namespace=ns,
                    query={
                        "top_k": 5,
                        "inputs": {
                            'text': query
                        }
                    }
                )

                for hit in results.get('result', {}).get('hits', []):
                    doc_id = hit.get('_id')
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_hits.append(hit)

    except Exception as e:
        logging.error(f"Error searching Pinecone index '{index_name}': {e}")
        return "", []

    if all_hits:
        sorted_hits = sorted(all_hits, key=lambda k: k.get('_score', 0), reverse=True)
        top_4_hits = sorted_hits[:4]

        all_contexts = []
        for hit in top_4_hits:
            # Store source information
            source_info = {
                'id': hit.get('_id'),
                'score': round(hit.get('_score', 0), 2),
                'page_number': hit.get('fields', {}).get('page_number', 'N/A'),
                'topic': hit.get('fields', {}).get('topic', 'N/A')
            }
            sources.append(source_info)
            
            formatted_result = (
                f"TEXT_CONTENT: {hit.get('fields', {}).get('chunk_text', 'N/A')}\n\n"
            )
            all_contexts.append(formatted_result)
        
        full_context = "\n".join(all_contexts)
        return full_context, sources
    else:
        return "", []

async def generate_quiz_from_context(context: str) -> list | None:
    """
    Generates a quiz from the provided context using the Gemini API.
    """
    system_prompt = f"""
    ->You are a quiz generation AI. Based on the provided context, create a multiple-choice quiz with 5 questions.
    ->The output must be a valid JSON array of objects, where each object has "question", "options" (an array of 4 strings), and "correct_option_id" (0-indexed integer).
    ->You must ask these questions as if you were coming up with them yourself, don't say "as stated in the text..." or "as mentioned in the text..." or anything like that.
    ->Each option(alternative to the quizes) must be less than a 100 characters long.
    **Context:**
    {context}

    **JSON Output:**
    """
    try:
        response = await model.generate_content_async(system_prompt)
        text_response = response.text.strip()
        json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            return json.loads(text_response)
    except Exception as e:
        logging.error(f"Error generating or parsing quiz from context: {e}")
        return None

# --- 3. TELEGRAM BOT HANDLERS ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "Welcome to the Advanced Quiz Bot!\n\n"
        "To start a quiz on a specific topic, use the command: /quiz <topic>\n"
        "For example: /quiz cells"
    )

async def quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Starts a quiz on a user-defined topic."""
    topic = " ".join(context.args)
    chat_id = None
    if update.message:
        chat_id = update.effective_chat.id
    elif update.callback_query:
        chat_id = update.callback_query.message.chat_id
    else:
        chat_id = context.user_data.get("last_chat_id")

    if not topic:
        if chat_id:
            await context.bot.send_message(chat_id, "Please provide a topic for the quiz. Usage: /quiz <topic>")
        return

    if chat_id:
        await context.bot.send_message(chat_id, f"Generating a quiz about '{topic}'. This might take a moment...")
    context.user_data["last_chat_id"] = chat_id

    # Store the topic for potential replay
    context.user_data["current_topic"] = topic

    retrieved_context, sources = await process_query_for_context(topic)
    if not retrieved_context:
        if chat_id:
            await context.bot.send_message(chat_id, "I'm sorry, I couldn't find enough information to create a quiz on that topic.")
        return

    # Store sources for later use
    context.user_data["sources"] = sources

    quiz_questions = await generate_quiz_from_context(retrieved_context)
    if not quiz_questions:
        if chat_id:
            await context.bot.send_message(chat_id, "I'm sorry, I was unable to generate a quiz. Please try another topic.")
        return
    context.user_data["quiz_questions"] = quiz_questions
    context.user_data["current_question"] = 0
    context.user_data["score"] = 0
    context.user_data["answers"] = []
    if chat_id:
        await send_question(chat_id, context)

async def send_question(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a new question to the user."""
    question_index = context.user_data.get("current_question", 0)
    quiz_questions = context.user_data.get("quiz_questions", [])
    
    if question_index < len(quiz_questions):
        question_data = quiz_questions[question_index]
        message = await context.bot.send_poll(
            chat_id=chat_id,
            question=question_data["question"],
            options=question_data["options"],
            type=Poll.QUIZ,
            correct_option_id=question_data["correct_option_id"],
            is_anonymous=False,
            explanation=f"The correct answer is {question_data['options'][question_data['correct_option_id']]}."
        )
        context.bot_data[message.poll.id] = {
            "chat_id": chat_id,
            "correct_option_id": question_data["correct_option_id"],
            "question": question_data["question"],
            "options": question_data["options"]
        }
        logging.info(f"Sent question to chat_id {chat_id}")

async def receive_poll_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Receive the poll answer and send the next question."""
    try:
        poll_id = update.poll_answer.poll_id
        quiz_data = context.bot_data[poll_id]
    except KeyError:
        logging.warning(f"Received answer for an unknown poll_id: {update.poll_answer.poll_id}")
        return

    user_answer_index = update.poll_answer.option_ids[0] if update.poll_answer.option_ids else None
    user_answer = quiz_data["options"][user_answer_index] if user_answer_index is not None else None
    correct = user_answer_index == quiz_data["correct_option_id"]
    context.user_data["answers"].append({
        "question": quiz_data["question"],
        "options": quiz_data["options"],
        "user_answer": user_answer,
        "user_answer_index": user_answer_index,
        "correct_option_id": quiz_data["correct_option_id"],
        "correct": correct
    })
    if correct:
        context.user_data["score"] = context.user_data.get("score", 0) + 1
    context.user_data["current_question"] = context.user_data.get("current_question", 0) + 1
    if context.user_data["current_question"] < len(context.user_data.get("quiz_questions", [])):
        await send_question(quiz_data["chat_id"], context)
    else:
        await show_result(quiz_data["chat_id"], context, update)

async def show_result(chat_id: int, context: ContextTypes.DEFAULT_TYPE, update: Update) -> None:
    score = context.user_data.get("score", 0)
    total_questions = len(context.user_data.get("quiz_questions", []))
    result_message = f"Quiz finished! Your final score is {score}/{total_questions}.\n\n"
    result_message += "This quiz was generated using information from the following sources:\n\n"
    sources = context.user_data.get("sources", [])
    for i, source in enumerate(sources, 1):
        result_message += (
            f"Source {i}:\nID: {source['id']}\nScore: {source['score']}\nPage Number: {source['page_number']}\nTopic: {source['topic']}\n\n"
        )
    keyboard = [
        [InlineKeyboardButton("Take Same Quiz Again", callback_data="replay_same"),
         InlineKeyboardButton("New Quiz on Same Topic", callback_data="new_same_topic")],
        [InlineKeyboardButton("Try Different Topic", callback_data="different_topic")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(chat_id=chat_id, text=result_message, reply_markup=reply_markup)
    # Log the quiz result
    user = update.effective_user
    quiz_data = {
        "score": score,
        "total_questions": total_questions,
        "answers": context.user_data.get("answers", []),
        "sources": sources
    }
    log_quiz_result(user.id, chat_id, user.username, quiz_data)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button presses from the inline keyboard."""
    query = update.callback_query
    await query.answer()  # Answer the callback query to remove the loading state

    if query.data == "replay_same":
        # Replay the exact same quiz
        context.user_data["current_question"] = 0
        context.user_data["score"] = 0
        await send_question(query.message.chat_id, context)
    
    elif query.data == "new_same_topic":
        # Generate a new quiz on the same topic
        topic = context.user_data.get("current_topic")
        if topic:
            # Clear the message with the buttons
            await query.message.delete()
            # Start a new quiz with the same topic
            context.args = topic.split()
            await quiz(update, context)
    
    elif query.data == "different_topic":
        # Prompt user to start a new quiz with a different topic
        await query.message.delete()
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Please use /quiz <topic> to start a new quiz on a different topic."
        )

def main() -> None:
    """Run the bot."""
    if not BOT_TOKEN:
        logging.error("No BOT_TOKEN found in environment variables!")
        return

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("quiz", quiz))
    application.add_handler(PollAnswerHandler(receive_poll_update))
    application.add_handler(CallbackQueryHandler(button_handler))

    print("Advanced Quiz Bot is running... Press Ctrl-C to stop.")
    application.run_polling()

if __name__ == "__main__":
    main()
