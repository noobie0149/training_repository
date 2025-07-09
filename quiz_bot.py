import logging
import os
import re
from dotenv import load_dotenv
from telegram import Update, Poll
from telegram.ext import Application, CommandHandler, PollAnswerHandler, ContextTypes
from pinecone import Pinecone
import google.generativeai as genai
from collections import deque
import json

# --- 1. SETUP AND INITIALIZATION ---

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Client Initialization ---
# Load credentials from .env file
BOT_TOKEN = os.getenv("attemptZeroBot_token")
PINECONE_API_KEY = os.getenv("pinecone_api")
GEMINI_API_KEY = os.getenv("gemma_gemini_api")

# Initialize Pinecone and Gemini
# Note: Your search logic might require a specific version of the pinecone-client library.
pc = Pinecone(api_key=PINECONE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# --- Pinecone and Gemini Model Setup ---
model = genai.GenerativeModel('gemini-2.5-flash')
index_name = "biology"
logging.info(f"Connected to Pinecone for index: {index_name}")
logging.info("Gemini Model 'gemini-1.5-flash' initialized.")


# --- 2. CORE LOGIC (PINECOME & GEMINI) ---

async def generate_search_queries(user_query: str) -> list[str]:
    """
    Generates conceptual search queries based on the user's topic.
    """
    # This is the query generation logic from your io_ot.py script.
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


async def process_query_for_context(user_query: str) -> str:
    """
    Orchestrates the query workflow to retrieve context from Pinecone using your specified method.
    """
    # 1. First, generate multiple, conceptual search queries.
    search_queries = await generate_search_queries(user_query)

    # 2. Search all specified namespaces and indices using the generated queries.
    dense_index = pc.Index(index_name)
    try:
        index_stats = dense_index.describe_index_stats()
        namespaces = list(index_stats.namespaces.keys())
    except Exception as e:
        logging.error(f"Could not connect to Pinecone index '{index_name}'. Error: {e}")
        return "I'm sorry, I'm having trouble connecting to my knowledge base right now."

    all_hits = []
    seen_ids = set() # Use a set to track unique document IDs and avoid duplicates

    try:
        # Loop through each generated query and search the database
        for query in search_queries:
            logging.info(f"Searching with conceptual query: '{query}'")
            for ns in namespaces:
                # This is the exact search logic you provided.
                results = dense_index.search(
                    namespace=ns,
                    query={
                        "top_k": 5, # Fetch a few results per conceptual query
                        "inputs": {
                            'text': query
                        }
                    }
                )

                for hit in results.get('result', {}).get('hits', []):
                    doc_id = hit.get('_id')
                    # If we haven't seen this document ID before, add its raw data
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_hits.append(hit)

    except Exception as e:
        logging.error(f"Error searching Pinecone index '{index_name}': {e}")
        return "An error occurred while searching the knowledge base."

    # --- Sort all collected hits by score and select the top 4 ---
    if all_hits:
        # Using _score as per your original script's logic
        sorted_hits = sorted(all_hits, key=lambda k: k.get('_score', 0), reverse=True)
        top_4_hits = sorted_hits[:4]
        logging.info(f"Selected top {len(top_4_hits)} hits based on scores.")

        all_contexts = []
        for hit in top_4_hits:
            # Using the exact field names from your original script
            formatted_result = (
                f"ID: {hit.get('_id')} | SCORE: {round(hit.get('_score', 0), 2)} | PAGE_NUMBER: {hit.get('fields', {}).get('page_number', 'N/A')}\n"
                f"TEXT_HEADER: {hit.get('fields', {}).get('topic', 'N/A')}\n"
                f"TEXT_CONTENT: {hit.get('fields', {}).get('chunk_text', 'N/A')}\n\n"
            )
            all_contexts.append(formatted_result)
        
        full_context = "\n".join(all_contexts)
        return full_context
    else:
        logging.warning("No context found from Pinecone search after multiple attempts.")
        return ""


async def generate_quiz_from_context(context: str) -> list | None:
    """
    Generates a quiz from the provided context using the Gemini API.
    """
    system_prompt = f"""
    You are a quiz generation AI. Based on the provided context, create a multiple-choice quiz with 5 questions.
    The output must be a valid JSON array of objects, where each object has "question", "options" (an array of 4 strings), and "correct_option_id" (0-indexed integer).
    You must ask these questions as if you were coming up with them yourself, don't say "as stated in the text..." or "as mentioned in the text..." or anything like that.

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
        "Welcome to the Dynamic Quiz Bot!\n\n"
        "To start a quiz on a specific topic, use the command: /quiz <topic>\n"
        "For example: /quiz cells"
    )

async def quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Starts a quiz on a user-defined topic."""
    topic = " ".join(context.args)
    if not topic:
        await update.message.reply_text("Please provide a topic for the quiz. Usage: /quiz <topic>")
        return

    await update.message.reply_text(f"Generating a quiz about '{topic}'. This might take a moment...")

    retrieved_context = await process_query_for_context(topic)
    if not retrieved_context:
        await update.message.reply_text("I'm sorry, I couldn't find enough information to create a quiz on that topic.")
        return

    quiz_questions = await generate_quiz_from_context(retrieved_context)
    if not quiz_questions:
        await update.message.reply_text("I'm sorry, I was unable to generate a quiz. Please try another topic.")
        return
        
    context.user_data["quiz_questions"] = quiz_questions
    context.user_data["current_question"] = 0
    context.user_data["score"] = 0
    
    await send_question(update.effective_chat.id, context)

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
            "correct_option_id": question_data["correct_option_id"]
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

    if update.poll_answer.option_ids and update.poll_answer.option_ids[0] == quiz_data["correct_option_id"]:
        context.user_data["score"] = context.user_data.get("score", 0) + 1

    context.user_data["current_question"] = context.user_data.get("current_question", 0) + 1

    if context.user_data["current_question"] < len(context.user_data.get("quiz_questions", [])):
        await send_question(quiz_data["chat_id"], context)
    else:
        await show_result(quiz_data["chat_id"], context)

async def show_result(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the final quiz result."""
    score = context.user_data.get("score", 0)
    total_questions = len(context.user_data.get("quiz_questions", []))
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"Quiz finished! Your final score is {score}/{total_questions}.\n\nSend /quiz <topic> to play again!"
    )
    context.user_data.clear()

def main() -> None:
    """Run the bot."""
    if not BOT_TOKEN:
        logging.error("No BOT_TOKEN found in environment variables!")
        return

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("quiz", quiz))
    application.add_handler(PollAnswerHandler(receive_poll_update))

    print("Bot is running... Press Ctrl-C to stop.")
    application.run_polling()

if __name__ == "__main__":
    main()
