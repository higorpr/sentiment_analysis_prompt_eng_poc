# Imports
from functions import (
    get_chat_messages,
    import_data,
    get_gpt_sentiment,
    generate_sentiment_label,
)


def main():
    # Get chat_id
    chat_id = input("Please enter chat_id\n")
    print(f"Analyzing chat: {chat_id}")

    # Retrieve messages:
    messages = get_chat_messages(chat_id)

    # Create messages dataframe:w
    messages_df = import_data(messages)

    # Display number of messages in chat:
    print(f"Number of messages in chat: {messages_df.shape[0]}\n")

    # Call chatGPT to calculate the average sentiment score:
    gpt_return = get_gpt_sentiment(messages_df)

    # Display chat sentiment
    print(f"The calculated whole chat score was {gpt_return['average']}")

    # Turn score to label:
    sentiment_label = generate_sentiment_label(gpt_return["average"])

    # Display chat sentiment label:
    print(f"O sentimento do usuário no chat enviado foi: {sentiment_label}")


if __name__ == "__main__":
    main()
