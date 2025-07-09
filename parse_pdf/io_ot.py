# gemini_qa_bot.py

import os
import logging
import re
from dotenv import load_dotenv
from telethon import TelegramClient, events
from pinecone import Pinecone
import google.generativeai as genai
from collections import deque

# --- 1. SETUP AND INITIALIZATION ---

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Client Initialization ---
# Load credentials from .env file
API_ID = os.getenv("telegram_id")
API_HASH = os.getenv("telegram_hash")
BOT_TOKEN = os.getenv("highschool_biology_bot")
PINECONE_API_KEY = os.getenv("pinecone_api")
GEMINI_API_KEY = os.getenv("gemma_gemini_api")

# Initialize all clients
client = TelegramClient('bot_session', API_ID, API_HASH).start(bot_token=BOT_TOKEN)
pc = Pinecone(api_key=PINECONE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# --- Pinecone and Gemini Model Setup ---
model = genai.GenerativeModel('gemma-3-27b-it')
index_name = "biology"
logging.info(f"Connected to Pinecone for index: {index_name}")
logging.info("Gemini Model 'gemma' initialized.")

# --- Conversation Memory ---
conversation_history = {}
HISTORY_LENGTH = 5 # The number of past messages to remember

# --- 2. QUERY AND ANSWER LOGIC ---

async def generate_search_queries(user_query: str) -> list[str]:
    """
    **MODIFIED**
    Dynamically generates conceptual search queries based on the alternatives in a multiple-choice question.
    The number of queries will match the number of options (e.g., A, B, C, D -> 4 queries).
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
    5.  ** The number of queries generated is dependant on the type of alternative questions, as shown below.
    Examples:
         {{"Question":"Which one of the following diseases is correctly matched with its symptoms and method of
           transmission?
                 A. Chickenpox - runny and stuffy nose - respiratory droplets
                 B. Measles - paralysis and hydrophobia - bite from infected animals
                 C. Polio - commonly no sign and symptoms - fecal-oral-route
                 D. Rabies - swollen and painful protid glands - through infected saliva          
           "queries generated" :
                 "1) what is chicken pox and how is it transmitted
                 2) what is measels and how is it transmitted
                 3) what is polio and how is it transmitted
                 4) what is rabies and how is it transmitted"}}
         {{"Question":"What happens to the substrate molecule when competitive inhibitor binds with an enzyme? It:
                 A. undergoes conformational change.
                 B. cannot bind with the active site.
                 C. binds to the allosteric site temporarily.
                 D. easily binds with the active site." ,
           "queries generated" :
                 1)what are competitive inhibitiors?
                 2)How do competitive inhibitor bind with enzymes?}}
         {{"Question":"What are cells?",
           "queries generated" :
                 "1)What are cells?"}}
         {{"Question":"what is inversion?",
           "queries generated" :
                 "1)What is inversion?"}}
    **User Question:** "{user_query}"

    **Conceptual Search Queries:**
    """
    try:
        response = await model.generate_content_async(system_prompt)
        # Split the response text by newlines and strip whitespace from each line
        queries = [query.strip() for query in response.text.strip().split('\n') if query.strip()]
        logging.info(f"Original query: '{user_query}' | Generated {len(queries)} conceptual queries: {queries}")
        # If no queries are generated, fall back to a simple keyword version of the original query
        if not queries:
             raise ValueError("Model failed to generate queries.")
        return queries
    except Exception as e:
        logging.error(f"Error during conceptual query generation: {e}")
        # Fallback for any failure: use the original query as a single search query
        return [user_query]

async def generate_final_answer(original_query: str, contents: str, history: str) -> str:
    """
    Generates the final answer based on the ORIGINAL query, retrieved context, and conversation history.
    The function is asynchronous.
    """
    system_prompt = f"""
    You are a specialized AI assistant. Your sole purpose is to answer the user's query based exclusively on the provided search results and conversation history. You must adhere to the following instructions without deviation.

    **Conversation History:**
    {history}

    **Instructions:**

    1.  **Analyze the User's Original Query:** The user wants to know: "{original_query}"

    2.  **Review the Provided Context:** You are given the following search results.
        ```context
        {contents}
        ```

    3.  **Synthesize the Answer:**
        * Formulate a descriptive answer to the user's original query using *only* the information found in the `TEXT_CONTENT` of the provided results and the conversation history.
        * Do not invent, infer, or use any information outside of the provided context.
        * Structure your answer to directly address the user's question format (e.g., for multiple choice, state the correct option and then explain why, for short-answer, provide a direct answer and explanation).
        * If you cannot find a suitable answer for "{original_query}", reply with "I'm sorry, there isn't a suitable answer to your question in the book."

    4.  **Cite Your Sources:**
        * After the answer, list the sources you used.
        * For each source, include its `ID`, `SCORE`, `TEXT_HEADER`, and `PAGE_NUMBER`.
        * Format each citation exactly as: `Source: \n ID: [ID], SCORE: [SCORE], HEADER: [TEXT_HEADER], PAGE_NUMBER: [PAGE_NUMBER]`

    **Output Mandate:**
    * Your entire output must consist of two parts ONLY: the synthesized answer first, followed by the list of source citations.
    * DO NOT add any introductory phrases, greetings, or concluding remarks.
    """
    try:
        response = await model.generate_content_async(system_prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error during final answer generation: {e}")
        return "There was an error generating the final answer."

# MODIFIED: Function signature now accepts chat_id
async def process_query(user_query: str, chat_id: int):
    """
    **MODIFIED**
    Orchestrates the new multi-query workflow and handles conversation history.
    """
    # 1. First, generate multiple, conceptual search queries.
    search_queries = await generate_search_queries(user_query)

    # 2. Search all specified namespaces and indices using the generated queries.
    index_name = "biology"
    dense_index = pc.Index(index_name)
    index_stats = dense_index.describe_index_stats()
    namespaces = list(index_stats.namespaces.keys())

    all_hits = []
    seen_ids = set() # Use a set to track unique document IDs and avoid duplicates

    try:
        # Loop through each generated query and search the database
        for query in search_queries:
            logging.info(f"Searching with conceptual query: '{query}'")
            for ns in namespaces:
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

    # --- Sort all collected hits by score and select the top 4 ---
    if all_hits:
        sorted_hits = sorted(all_hits, key=lambda k: k.get('_score', 0), reverse=True)
        top_4_hits = sorted_hits[:4]
        logging.info(f"Selected top {len(top_4_hits)} hits based on scores.")

        all_contexts = []
        for hit in top_4_hits:
            formatted_result = (
                f"ID: {hit.get('_id')} | SCORE: {round(hit.get('_score', 0), 2)} | PAGE_NUMBER: {hit.get('fields', {}).get('page_number', 'N/A')}\n"
                f"TEXT_HEADER: {hit.get('fields', {}).get('topic', 'N/A')}\n"
                f"TEXT_CONTENT: {hit.get('fields', {}).get('chunk_text', 'N/A')}\n\n"
            )
            all_contexts.append(formatted_result)
        
        full_context = "\n".join(all_contexts)
    else:
        full_context = ""
    print(all_contexts)
    if not full_context.strip():
        logging.warning("No context found from Pinecone search after multiple attempts.")
        return "I'm sorry, I couldn't find any relevant information in the book to answer your question."

    # 3. Get history, generate final answer, and update history
    user_history = conversation_history.get(chat_id, deque(maxlen=HISTORY_LENGTH))
    history_str = "\n".join([f"User: {q}\nBot: {a}" for q, a in user_history])
    
    # MODIFIED: Pass history_str to the answer generation function
    final_answer = await generate_final_answer(user_query, full_context, history_str)
    
    # Update conversation history with the new exchange
    if "I'm sorry" not in final_answer:
        user_history.append((user_query, final_answer))
        conversation_history[chat_id] = user_history

    return final_answer


# --- 3. TELEGRAM BOT HANDLERS ---

@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    """Handles the /start command."""
    chat_id = event.chat_id
    # Clear history for the user on /start
    if chat_id in conversation_history:
        del conversation_history[chat_id]
        logging.info(f"Cleared conversation history for chat_id: {chat_id}")

    welcome_message = (
        "Hello! I am a Q&A bot for the Grades 9,10,11 & 12 Biology curriculum.\n\n"
        "Please ask me a question, and I will find the answer for you from the textbook."
    )
    await event.respond(welcome_message)
    logging.info(f"Started new session for chat_id: {chat_id}")


@client.on(events.NewMessage)
async def message_handler(event):
    """Handles all non-command text messages."""
    if event.text.startswith('/'):
        return

    user_query = event.text
    chat_id = event.chat_id
    username = event.sender.username if event.sender and hasattr(event.sender, 'username') else None
    logging.info(f"Received query from chat_id {chat_id}: '{user_query}'")

    async with client.action(chat_id, 'typing'):
        try:
            response_text = await process_query(user_query, chat_id)
            await event.respond(response_text)
            logging.info(f"Successfully sent response to chat_id {chat_id}")
            # Log only username, chat_id, user_query, and response_text
            log_entry = {
                "username": username,
                "chat_id": chat_id,
                "query": user_query,
                "response": response_text
            }
            with open("bot.log", "a") as f:
                import json
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.error(f"An error occurred in message_handler for chat_id {chat_id}: {e}")
            await event.respond("I'm sorry, an unexpected error occurred. Please try again later.")


# --- 4. MAIN EXECUTION BLOCK ---

async def main():
    """Main function to run the bot."""
    logging.info("Bot is starting up...")
    await client.run_until_disconnected()
    logging.info("Bot has stopped.")

if __name__ == '__main__':
    client.loop.run_until_complete(main())